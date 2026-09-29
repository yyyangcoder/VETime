# -*- coding: utf-8 -*-
# License: Apache-2.0 License

import sys

import pandas as pd
import torch
import random, argparse, os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (TSB-AD-main)
parent_dir = os.path.dirname(current_dir)

# 将父目录加入系统路径
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.evaluation.metrics import get_metrics_pred
from TSB_AD.evaluation.evaluator import Evaluator
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

PASS_LIST = [
    "Daphnet", "CATSv2", "SWaT", "LTDB", "TAO", "Exathlon", "MITDB", "MSL", "SMAP", "SMD", "SVDB", "OPP",
    "Stock", "IOPS", "MGAB", "NEK", "Power", "SED", "TODS", "UCR"
]
USE_LIST = ["NAB", "WSD", "YAHOO"]
PASS_else_LIST = ["Stock", "IOPS", "MGAB", "NEK", "Power", "SED", "TODS", "UCR", "YAHOO"]
USE_else_LIST = ["GHL","Daphnet", "CATSv2", "SWaT", "LTDB", "TAO", "Exathlon", "MITDB", "MSL", "SMAP", "SMD", "SVDB", "OPP","Genesis","PSM","CredicCard","GECCO"]
TARGET_LIST = ["GHL", "Daphnet", "Exathlon", "Genesis", "GECCO", "MITDB", "SVDB", "LTDB", "CATSv2", "CATS", "TAO"]

DATA_INIT_SETTING = {
    "img_size": 224,
    "T_sqrt": False,
}

DATASET_FILTERS = {
    'PASS': PASS_LIST,
    'USE': USE_LIST,
    'PASS_else': PASS_else_LIST,
    'USE_else': USE_else_LIST,
    'TARGET': TARGET_LIST,
}
DEFAULT_DATA_SETTING = DATA_INIT_SETTING
DEFAULT_DATASET_SETTING = USE_else_LIST

# 设置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())

from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

# determine sliding window (period) based on ACF
# … keep same as previous

def find_length_rank(data, rank=1):
    data = data.squeeze()
    if len(data.shape)>1: return 0
    if rank==0: return 1
    data = data[:min(20000, len(data))]
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]
        max_local_max = sorted_local_max[0]
        if rank == 1: max_local_max = sorted_local_max[0]
        if rank == 2:
            for i in sorted_local_max[1:]:
                if i > sorted_local_max[0]:
                    max_local_max = i
                    break
        if rank == 3:
            for i in sorted_local_max[1:]:
                if i > sorted_local_max[0]:
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]:
                    max_local_max = i
                    break
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125


def find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125


def run_one_file(args, filename, data_setting=DEFAULT_DATA_SETTING):
    if not any(filter_item in filename for filter_item in args.dataset_setting):
        print(f"Skipping {filename} due to dataset_setting filter")
        return None

    file_path = os.path.join(args.data_direc, filename)
    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    slidingWindow = find_length_rank(data, rank=1)
    train_index = filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    model_name = args.AD_Name

    Optimal_Det_HP = Optimal_Uni_algo_HP_dict.get(model_name, {})

    if model_name in Semisupervise_AD_Pool:
        output = run_Semisupervise_AD(model_name, data_train, data, **Optimal_Det_HP)
    elif model_name in Unsupervise_AD_Pool:
        output = run_Unsupervise_AD(model_name, data, **Optimal_Det_HP)
    else:
        raise Exception(f"{args.AD_Name} is not defined")

    if not isinstance(output, np.ndarray):
        print(f'At {filename}: {output}')
        return None

    output_norm = MinMaxScaler(feature_range=(0,1)).fit_transform(output.reshape(-1,1)).ravel()
    
    save_path = os.path.join(args.output_dir, 'eval_spot', filename.split('.')[0])
    evaluator = Evaluator(label, output_norm, save_path)
    thresholds = evaluator.find_thres(method='spot', init_score=0.2, q=[1e-4], verbose=False)
    pred_threshold = thresholds[0]
    
    pred = output_norm > pred_threshold
    eval_result = get_metrics_pred(output_norm, label, slidingWindow=slidingWindow, pred=pred)

    if args.save:
        score_filename = f"{os.path.splitext(filename)[0]}_{args.AD_Name}_score.csv"
        score_path = os.path.join(args.output_dir, score_filename)
        pd.DataFrame({'score': output_norm, 'pred': pred.astype(int), 'label': label}).to_csv(score_path, index=False)

    return {'filename': filename, 'AD_Name': args.AD_Name, 'slidingWindow': slidingWindow, **eval_result}


if __name__ == '__main__':
    import csv
    parser = argparse.ArgumentParser(description='Running TSB-AD')
    parser.add_argument('--data_direc', type=str, default='//Datasets/TSB-AD-M/DADA/data', help='Folder containing csv data files')
    parser.add_argument('--AD_Name', type=str, default='TimesFM', help='Model name in pool, e.g., IForest, LOF, SAND...')
    parser.add_argument('--output_dir', type=str, default=f'//results_TimesFM_final', help='Directory to save CSV output')
    parser.add_argument('--save', type=bool, default=False, help='Whether to save per-dataset score/pred files')
    parser.add_argument('--file_list', type=str, default='', help='Optional comma-separated list of files in data_direc to process; if empty, process all .csv')
    parser.add_argument('--dataset_mode', type=str, default='TARGET', help='Which dataset filter list to apply from Test_TSB_file')
    parser.add_argument('--data_setting', type=str, default='DATA_INIT_SETTING', help='Currently only DATA_INIT_SETTING is supported')
    args = parser.parse_args()

    args.dataset_setting = DATASET_FILTERS.get(args.dataset_mode, DEFAULT_DATASET_SETTING)
    args.data_setting = DEFAULT_DATA_SETTING

    os.makedirs(args.output_dir, exist_ok=True)

    if args.file_list.strip():
        files = [f.strip() for f in args.file_list.split(',') if f.strip()]
    else:
        files = [f for f in os.listdir(args.data_direc) if f.endswith('.csv')]
    output_csv = os.path.join(args.output_dir, f"result_{args.AD_Name}.csv")
    
    file_exists = os.path.isfile(output_csv)
    print(f"--- 结果将实时追加保存至: {output_csv} ---")
    for filename in files:
        print(f'\n>>> Running {filename} with AD_Name {args.AD_Name}')
        record = run_one_file(args, filename)
        
        if record is not None:
            try:
                # 使用 'a' (append) 模式打开文件，如果文件不存在会自动创建
                with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                    # 使用 DictWriter 根据字典的 key 写入数据
                    writer = csv.DictWriter(f, fieldnames=record.keys())
                    
                    # 如果文件是新建的（第一次循环），先写入表头
                    if not file_exists:
                        writer.writeheader()
                        file_exists = True # 标记文件已创建，后续循环不再写表头
                    
                    # 写入当前这一行的数据
                    writer.writerow(record)
                    print(f"✅ 已保存结果: {filename}")
            except Exception as e:
                print(f"❌ 保存文件 {filename} 时出错: {e}")
        else:
            print(f"⏭️ 跳过: {filename} (无有效结果)")
    # records = []
    # for filename in files:
    #     print('Running', filename, 'with AD_Name', args.AD_Name)
    #     record = run_one_file(args, filename)
    #     if record is not None:
    #         records.append(record)

    # if records:
    #     out_df = pd.DataFrame(records)

    #     def get_dataset_name(filename):
    #         parts = os.path.splitext(filename)[0].split('_')
    #         if parts[0].isdigit():
    #             return parts[1] if len(parts) > 1 else parts[0]
    #         else:
    #             return parts[0]

    #     out_df['dataset_group'] = out_df['filename'].apply(get_dataset_name)

    #     group_mean = out_df.groupby('dataset_group').mean(numeric_only=True).reset_index()
    #     group_mean_rows = []
    #     for _, g in group_mean.iterrows():
    #         row = {'filename': f"Mean_{g['dataset_group']}", 'AD_Name': args.AD_Name, 'slidingWindow': ''}
    #         numeric_cols = [c for c in g.index if c not in ['dataset_group']]
    #         row.update({c: g[c] for c in numeric_cols})
    #         group_mean_rows.append(row)

    #     out_df = out_df.drop(columns=['dataset_group'])
    #     out_df = pd.concat([out_df, pd.DataFrame(group_mean_rows)], ignore_index=True)

    #     output_csv = os.path.join(args.output_dir, f"result_{args.AD_Name}.csv")
    #     out_df.to_csv(output_csv, index=False)

    # print("Done.")

    # 去重：对结果 CSV 按 filename 去重，保留第一条
    print(f"\n正在对结果文件去重...")
    dedup_csv_path = output_csv
    try:
        with open(dedup_csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
            rows = [row for row in reader if row]
        
        if len(rows) > 0:
            filename_idx = fieldnames.index('filename')
            seen = set()
            dedup_rows = []
            for row in rows:
                filename = row[filename_idx]
                if filename not in seen:
                    seen.add(filename)
                    dedup_rows.append(row)
            
            if len(dedup_rows) < len(rows):
                with open(dedup_csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(fieldnames)
                    writer.writerows(dedup_rows)
                print(f"✅ 去重完成：{len(rows)} -> {len(dedup_rows)} (删除 {len(rows) - len(dedup_rows)} 条重复记录)")
            else:
                print(f"✅ 无需去重：共 {len(dedup_rows)} 条记录")
    except Exception as e:
        print(f"⚠️ 去重时出错：{e}")

    print("\nDone.")
