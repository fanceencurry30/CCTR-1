import os
import json
import torch
import numpy as np
import random
from tqdm import tqdm

def prepare_data(json_folder, output_file, seed=42):
    """
    将多个JSON文件中的数据合并、处理并保存为单一的PyTorch二进制文件，
    以便快速、高效地进行内存映射加载。数据经过筛选以创建1:1的均衡数据集。
    
    :param json_folder: 包含JSON文件的文件夹路径
    :param output_file: 输出的PyTorch二进制文件路径
    :param seed: 随机种子，用于可重现性
    """
    print(f"开始处理源文件夹: {json_folder}")
    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]
    
    all_ocr_c100 = []
    all_lm_c100 = []
    all_gt_c100 = []
    
    # 1. 遍历所有JSON文件，加载并展开数据
    for file_path in tqdm(json_files, desc="处理JSON文件进度", position=0):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 为文件内部处理添加一个内层进度条
            for sample in tqdm(data, desc=f"处理 {os.path.basename(file_path)}", position=1, leave=False):
                if 'gt_c100' in sample:
                    # 将序列中的每个位置作为一个独立的样本添加
                    all_ocr_c100.extend(sample['ocr_c100'])
                    all_lm_c100.extend(sample['lm_c100'])
                    all_gt_c100.extend(sample['gt_c100'])

    print(f"数据处理完成。总样本数: {len(all_gt_c100)}")
    
    # 2. 分类样本为"match"和"mismatch"
    match_samples = []
    mismatch_samples = []
    
    for i in range(len(all_gt_c100)):
        ocr_top1 = np.argmax(all_ocr_c100[i])
        gt_top1 = np.argmax(all_gt_c100[i])
        if ocr_top1 == gt_top1:
            match_samples.append(i)
        else:
            mismatch_samples.append(i)
    
    num_mismatch = len(mismatch_samples)
    print(f"不匹配样本数: {num_mismatch}")
    print(f"匹配样本数: {len(match_samples)}")
    
    # 从"match"样本中随机选择num_mismatch个样本
    random.seed(seed)
    selected_match_samples = random.sample(match_samples, 2*num_mismatch)
    
    # 4. 构建均衡数据集
    #selected_indices = selected_match_samples + mismatch_samples  #均衡样本
    selected_indices = match_samples + mismatch_samples   #全样本
    #selected_indices = mismatch_samples                  #负样本
    selected_ocr_c100 = [all_ocr_c100[i] for i in selected_indices]
    selected_lm_c100 = [all_lm_c100[i] for i in selected_indices]
    selected_gt_c100 = [all_gt_c100[i] for i in selected_indices]
    
    print(f"均衡数据集样本数: {len(selected_gt_c100)}")
    
    # 5. 将数据分块转换为PyTorch张量，并显示进度
    print("正在将数据转换为张量...")
    
    def list_to_tensor_with_progress(data_list, desc):
        chunks = []
        chunk_size = 100000  # 每次处理10万个样本
        for i in tqdm(range(0, len(data_list), chunk_size), desc=desc):
            chunk = data_list[i:i + chunk_size]
            chunks.append(torch.tensor(chunk, dtype=torch.float32))
        return torch.cat(chunks, dim=0).unsqueeze(1)

    ocr_tensor = list_to_tensor_with_progress(selected_ocr_c100, "转换 OCR 数据")
    lm_tensor = list_to_tensor_with_progress(selected_lm_c100, "转换 LM 数据")
    gt_tensor = list_to_tensor_with_progress(selected_gt_c100, "转换 GT 数据")
    
    # 6. 将所有张量保存在一个字典中
    data_to_save = {
        'ocr_c100': ocr_tensor,
        'lm_c100': lm_tensor,
        'gt_c100': gt_tensor,
    }
    
    # 7. 保存到文件
    print(f"正在将处理好的数据保存到: {output_file}")
    torch.save(data_to_save, output_file)
    print("数据预处理完成！")

if __name__ == "__main__":
    JSON_FOLDER = './output_train/svtr'
    OUTPUT_FILE = 'final_all_last_svtr_p30.pt'
    
    if not os.path.exists(OUTPUT_FILE):
        prepare_data(JSON_FOLDER, OUTPUT_FILE)
    else:
        print(f"'{OUTPUT_FILE}' 已存在，跳过预处理。如果需要重新处理，请删除该文件。")
        