import math
import os
import einops
from peft import LoraConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Vision_encoder.Vit4AD import MAETS_AD, VitTS_AD
# from loss.head import AD_classifier,TS_Reconstruction,AD_Reconstruction_P

import os
# 获取当前文件所在目录的父目录（项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
vision_PATH = os.path.join(BASE_DIR, 'checkpoints', 'weight_v')

class V_model(nn.Module):
    def __init__(self, vision_name=None,unpatch=True,MAX_L=5000,finetune_type='ln',**kwargs):
        super().__init__() 
        
        vision_weight = os.path.join(vision_PATH,vision_name)
        if 'vit' in vision_name:
            self.encode_image= VitTS_AD(vision_weight)
            self.config = self.encode_image.config
            self.patch_size = self.config.patch_size
            self.hidden_size = self.config.hidden_size
        elif 'mae' in vision_name:
            self.encode_image= MAETS_AD(vision_weight)
            self.config = self.encode_image.config
            self.patch_size = self.config['patch_size']
            self.hidden_size = self.config['embed_dim']

        self.MAX_L=MAX_L
        self._freeze_layers() 

        if finetune_type != 'full':
            for n, param in self.encode_image.named_parameters():
                if 'ln' == finetune_type:
                    param.requires_grad = 'norm' in n
                elif 'bias' == finetune_type:
                    param.requires_grad = 'bias' in n
                elif 'none' == finetune_type:
                    param.requires_grad = False
                elif 'mlp' in finetune_type:
                    param.requires_grad = '.mlp.' in n
                elif 'attn' in finetune_type:
                    param.requires_grad = '.attn.' in n

    def _freeze_layers(self):
        # freeze all the parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def fold_image(self,images, P_L,p_values, img_size=224,T_sqrt=False):
        B, C, Num, W = images.shape
        results_2d = []
        results_size_out = []
        step_size = self.patch_size
        for b in range(B):
            P = int(P_L[b])
            image=images[b]
            p_value=p_values[b]
            T = W // step_size

            if T_sqrt:
                T_p = max(int(round(math.sqrt(T))), 1)
            else:
                T_p = P // step_size if P// step_size  > 1 else max(int(round(math.sqrt(T))), 1)

            init_h,init_w = T//T_p, T_p
            pad_patch = 0
            if T % T_p != 0:
                pad_patch = (T_p - T % T_p)
                pad_pixels = pad_patch * step_size #2*patchsize
                img_pad = F.pad(image.unsqueeze(0), (0, pad_pixels, 0, 0), "constant", 0).squeeze(0)
                init_h += 1

                img_pad[:, :, W:] = p_value.T[:, :, None]

            else:
                img_pad = image
            img_2d_l=[]
            for j in range(Num):
                img_pad_j=img_pad[:,j,:]
                parts = [img_pad_j[:,i*T_p*step_size:(i+1)*T_p*step_size].unsqueeze(1) for i in range(init_h)]
                img_2d_j = torch.cat(parts,axis=1)
                img_2d_l.append(img_2d_j)
            img_2d = torch.cat(img_2d_l,axis=1)
            img_resized_y = F.interpolate(img_2d.unsqueeze(0), size=(img_size, img_2d.shape[2]), mode='nearest', align_corners=None)
            img_final = F.interpolate(img_resized_y, size=(img_size, img_size), mode='bilinear', align_corners=False)
            results_2d.append(img_final)
            img_size_out = [init_h, init_w, pad_patch, img_size, Num]
            results_size_out.append(img_size_out)

        img_2d_batch = torch.cat(results_2d, dim=0)
        return img_2d_batch,results_size_out

    def unfold_image(self, x0,size):
        B, L, D = x0.shape
        
        recovered_list = []
        for i in range(B):
            x = x0[i]
            init_h,init_w,pad,h,Num = map(int, size[i])
            w=h = h//self.patch_size
            
            assert h * w == L
            output = x.transpose(0, 1).view(D, h, w)
            output = F.adaptive_avg_pool2d(output.unsqueeze(0), (init_h*Num, w)).squeeze(0)

            output_up = F.interpolate(
                output.unsqueeze(0), 
                size=(init_h*Num, init_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            patches=[]
            for j in range(Num):
                patch=output_up[:, init_h*j:init_h*(j+1), :].view(D, -1).contiguous()
                if pad>0:
                    patch = patch[:,:-pad]
                patches.append(patch)
            unfold = torch.cat(patches, dim=-1).transpose(0, 1)
            
            recovered_list.append(unfold)
        recovered_list = torch.stack(recovered_list)
        return recovered_list

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                The final hidden states of the model.
            grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
                The temporal, height and width of feature shape of each image in LLM.

        Returns:
            `torch.Tensor`: hidden_states.
        """
        hidden_states = self.encode_image(hidden_states)
        output_ts_patch =hidden_states[:,1:,:]

        return output_ts_patch,None