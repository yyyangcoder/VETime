# train_ad_qwen_vl.py
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from Test_TSB import PASS_LIST,TSB_test
from evaluation.metrics import fast_get_metrics
from model.Vision_encoder.V_encoder import V_model
from loss.loss import load_balance_loss
from model.TS_encoder.ts_model import TS_Model
from model.TS_encoder.config import default_config_t
from dataset.dataloader import AnomalyDataset,collate_fn
import logging
from tqdm.auto import tqdm
import os
from datetime import datetime
from model.VETime import VETIME
from Test_TSB import EarlyStopping
from functools import partial
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ["TRANSFORMERS_VERBOSITY"] = "error" 
import torch
torch.cuda.empty_cache()
def main(args):
    accelerator = Accelerator(
        mixed_precision="bf16",          
        gradient_accumulation_steps=4,    
        log_with="tensorboard",           
        project_dir="./output/logs"
    )

    logger.info(f"Using {accelerator.num_processes} {'GPUs' if accelerator.num_processes > 1 else 'CPU'}")

    vision_model = V_model(args.vision_name,unpatch=True)
    config_v = vision_model.config
    if 'mae' in args.vision_name:
        patch_size=config_v['patch_size']
    else:
        patch_size=config_v.patch_size

    ts_model = TS_Model(default_config_t) 
    if args.ts_path!=None:
        state_ts_dict = torch.load(args.ts_path, map_location='cpu')['model_state_dict']
        ts_model.load_state_dict(state_ts_dict)

    model = VETIME(config_v,vision_model,default_config_t,ts_model,args.model_name)  
    if args.vetime_path!=None:
        state_dict = torch.load(args.vetime_path, map_location='cpu')
        model.load_state_dict(state_dict)
    del vision_model,ts_model

    collatefn = partial(collate_fn, patch_size=patch_size)
    train_dataset = AnomalyDataset(args.dataset_path, patch_size=patch_size, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=collatefn, shuffle=False, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    

    trainable_params = (param for param in model.parameters() if param.requires_grad)

    optimizer = torch.optim.Adam(trainable_params,lr=1e-4, weight_decay=1e-2)

    model, optimizer,train_loader  = accelerator.prepare(
        model, optimizer, train_loader
    )

    model.train()
    global_step = 0
    epochs = args.num_epochs
    output = []
    device = accelerator.device
    data_setting=args.data_setting
    img_size=data_setting['img_size']
    name_save=f'./output/{args.model_name}__{img_size}_best.pth'
    
    early_stopping = EarlyStopping(patience=4, verbose=True, path=name_save)            
    output_path0=f'./output/score/uni/{args.model_name}_train'
    os.makedirs(output_path0, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_probs,all_preds, all_labels = [], [], []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}[Train]", disable=not accelerator.is_local_main_process)
        j=0
        for batch in progress_bar:
            labels = batch["labels"]
            images = batch["image"]  # (B, C, H, W)
            time_series, att_mask = batch['time_series'],batch['attention_mask']
            mask = batch['mask']
            period = batch['period']
            p_value = batch['padding_value']

            if labels.shape[1]>model.MAX_L:
                data_splits = model.split_data(images, time_series, att_mask, labels)
                loss1=0
                loss2=0
                logits_list=[]
                for data_part in data_splits:
                    img_part, ts_part, att_mask, label_part = data_part
                    images, init_img_size = model.vit_encoder.fold_image(img_part, period,p_value ,**data_setting)

                    local_embeddings1, m_w, loss_cl,local_embeddings2 = model(images, ts_part, att_mask, init_img_size,label_part)

                    loss01, logit  = model.anomaly_detection_loss(local_embeddings1, label_part)

                    loss02, rec= model.weighted_reconstruction_loss(local_embeddings2, ts_part, att_mask, label_part)
                    loss2=loss2+loss02
                    loss2 = loss2 + 0.1*loss_cl + 0.2*load_balance_loss(m_w)
                    loss1 = loss1+loss01
                    logits_list.append(logit)

                logits = torch.cat(logits_list, dim=1)

            else:

                images, init_img_size = model.vit_encoder.fold_image(images, period,p_value ,**data_setting)

                local_embeddings1, m_w, loss_cl,local_embeddings2 = model(images, time_series, att_mask, init_img_size,labels)

                loss1, logits  = model.anomaly_detection_loss(local_embeddings1, labels)

                loss2, rec= model.weighted_reconstruction_loss(local_embeddings2, time_series, att_mask, labels)
                
                loss2 = loss2 + 0.2*load_balance_loss(m_w)+0.1*loss_cl
            accelerator.backward(loss1+loss2)

            global_step+=1
            if global_step % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss1.item()+loss2.item()
            progress_bar.set_postfix({"loss": total_loss})

            probs = torch.softmax(logits, dim=-1)[:, :,1]
            preds = (probs > 0.5).float()

            probs,preds, labels = accelerator.gather_for_metrics((probs,preds, labels))
            j+=1

            for i in range(probs.shape[0]):
                all_probs.append(probs[i].detach().cpu().numpy().reshape(-1))
                all_preds.append(preds[i].detach().cpu().numpy().reshape(-1))
                all_labels.append(labels[i].detach().cpu().numpy().reshape(-1).astype(int))
            
            del images,logits, loss1, probs, preds, labels, loss2 
            torch.cuda.empty_cache()
        all_probs = np.concatenate(all_probs)  # (total_points,)
        all_preds= np.concatenate(all_preds) 
        all_labels = np.concatenate(all_labels)
        if np.any(np.isnan(all_probs)):
            print("⚠️ Warning: all_probs contains NaN values!")
        train_metrics = fast_get_metrics(all_probs, all_labels)
        
        avg_train_loss = total_loss / len(train_loader)
        accelerator.log({"epoch_train_loss": avg_train_loss}, step=epoch)
        print(f"\n[Epoch {epoch + 1}/{epochs}] 🟩 Training Summary:")
        print(f"  Avg Loss: {avg_train_loss:.4f}")
        for k, v in train_metrics.items():
            print(f"  Train {k}: {v:.4f}")
        if (epoch+1) % 2 == 0 or epoch == epochs - 1:
            model.eval()
            avg_val_loss=TSB_test(model,args.model_name,args.data_setting,device,dataset_setting=PASS_LIST)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            timestamp = datetime.now().strftime("%m%d-%H")
            name_save=f'./output/{args.model_name}__{img_size}_{avg_val_loss:.4f}_{timestamp}.pth'
                        
            torch.save(unwrapped_model.state_dict(), name_save)
            logger.info(f"Best model saved at epoch {epoch+1} with val_loss={avg_val_loss:.4f}")
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "train_metrics": {k: round(v, 6) for k, v in train_metrics.items()},
                "val_loss": round(avg_val_loss, 6) if avg_val_loss is not None else None,
            }
            output.append(epoch_log)
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
            
    loss_all=TSB_test(model,args.model_name,args.data_setting,device,dataset_setting=PASS_LIST)
    print(loss_all)
    accelerator.end_training()
    logger.info("Training completed!")

    return output

if __name__ == "__main__":
    DATA_INIT_SETTING = {
    "img_size": 224,
    "T_sqrt":  False,
    }
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset_path', default='./dataset'
                        , type=str, help='Path to the data file')
    parser.add_argument('--dataset_test_dir', type=str, default='./dataset/TSB-AD/Datasets/TSB-AD-U')
    parser.add_argument('--file_list', type=str, default='./dataset/TSB-AD/Datasets/File_List/TSB-AD-U.csv')
    parser.add_argument('--model_name', default= 'VETime', type=str, help='Name of the model')
    parser.add_argument('--seed', type=int, default=64, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of data loader workers')
    parser.add_argument('--num_epochs', type=int, default=4, help='epochs number')
    parser.add_argument('--output_file_path', default='./output/result.json',type=str, help='Path to the output file')
    parser.add_argument('--keep_idx_path', type=str, required=False, help='Path to the keep idx file')
    parser.add_argument('--device', type=str, default='auto', help='Device to use for evaluation')
    parser.add_argument('--data_setting', type=str, default=DATA_INIT_SETTING, help='Device to use for evaluation')
    parser.add_argument('--vision_path', type=str, default='./checkpoints/weight_v'
                        , help='vision_weight')
    parser.add_argument('--ts_path', type=str, default=None
                        , help='TS_weight')
    parser.add_argument('--vetime_path', type=str, default=None
                        , help='VETime_weight')
    parser.add_argument('--vision_name', type=str, default='mae_visualize_base.pth'
                        , help='vision_weight_name')
    
    args = parser.parse_args()
    output_file_path = args.output_file_path.replace('result.json', f'{args.model_name.replace("/", "-")}_result.json')

    results = main(args)

    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)