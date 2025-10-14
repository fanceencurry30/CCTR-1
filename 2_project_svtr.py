import json
import os
import torch
import numpy as np
import multiprocessing
from function import ProbabilityGenerator
from tqdm import tqdm
import re


def normalize_symbols(text):
    text = re.sub(r'[【】]', lambda x: '[' if x.group(0) == '【' else ']', text)
    text = re.sub(r'[:：]', ':', text)
    text = re.sub(r'[，,]', ',', text)
    text = text.lower()
    text = text.translate(str.maketrans(
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
        'abcdefghijklmnopqrstuvwxyz'
    ))
    return text

def load_mappings(map_file, ocr_char_file):
    with open(map_file, 'r', encoding='utf-8') as f:
        ocr_to_lm = json.load(f)
    with open(ocr_char_file, 'r', encoding='utf-8') as f:
        ocr_chars = [line.strip() for line in f.readlines()]
    return ocr_to_lm, ocr_chars

def filter_c100(topk_indices, topk_probs, is_ctc, skip, x):
    filtered_indices = []
    filtered_probs = []
    prev_raw_idx = None
    for t in range(len(topk_indices)):
        top1_idx = topk_indices[t][0]
        if is_ctc:
            if top1_idx == 0:
                prev_raw_idx = top1_idx
                continue
            if skip and prev_raw_idx is not None and top1_idx == prev_raw_idx:
                prev_raw_idx = top1_idx
                continue
            prev_raw_idx = top1_idx
        else:
            if top1_idx == 0:
                break
            if not (1 <= top1_idx <= x):
                continue
        filtered_indices.append(topk_indices[t])
        filtered_probs.append(topk_probs[t])
    return filtered_indices, filtered_probs

def generate_gt_c100(true_text, ocr_chars, topk_indices):
    gt_c100_list = []
    for t, char in enumerate(true_text):
        if char not in ocr_chars:
            return None
        ocr_idx = ocr_chars.index(char) + 1
        if ocr_idx not in topk_indices[t]:
            return None
        gt_probs = [1.0 if idx == ocr_idx else 0.0 for idx in topk_indices[t]]
        gt_c100_list.append(gt_probs)
    return gt_c100_list

def extract_decoded_text(decoded_item):
    if isinstance(decoded_item, list):
        return decoded_item[0] if len(decoded_item) > 0 else ""
    elif isinstance(decoded_item, str):
        return decoded_item
    else:
        raise ValueError(f"Unexpected type for decoded_item: {type(decoded_item)}")

def process_single_file_worker(args):
    json_file_path, gpu_id, map_file, ocr_char_file, lm_model_path, is_ctc, skip = args
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    ocr_to_lm, ocr_chars = load_mappings(map_file, ocr_char_file)
    x = len(ocr_chars)
    lm_model = ProbabilityGenerator(lm_model_path)
    
    local_train_data = []
    stats = {
        'total_samples': 0,
        'filtered_samples': 0,
        'filtered_correct': 0,
        'retained_samples': 0,
        'retained_correct': 0
    }
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    topk_probs = data['topk_probs']
    topk_indices = data['topk_indices']
    labels = data['labels']
    decoded_texts = data['decoded_texts']

    for sample_idx in range(len(decoded_texts)):
        stats['total_samples'] += 1
        true_text = labels[sample_idx][0] if isinstance(labels[sample_idx], list) else labels[sample_idx]
        decoded_item = decoded_texts[sample_idx]
        decoded_text = extract_decoded_text(decoded_item)

        if not isinstance(decoded_text, str) or not isinstance(true_text, str):
            stats['filtered_samples'] += 1
            continue

        norm_true_text = normalize_symbols(true_text)
        norm_decoded_text = normalize_symbols(decoded_text)
        true_text = norm_true_text
        is_correct = norm_true_text == norm_decoded_text

        true_len = len(norm_true_text)
        decoded_len = len(norm_decoded_text)
        
        # --- 新增修改：检查解码文本是否为空 ---
        if decoded_len == 0:
            stats['filtered_samples'] += 1
            if is_correct: # 如果真实文本也为空，则算正确
                stats['filtered_correct'] += 1
            continue
        # --- 结束修改 ---

        if true_len == 1 or (true_len > 1 and norm_true_text[0] != norm_decoded_text[0]):
            stats['filtered_samples'] += 1
            if is_correct:
                stats['filtered_correct'] += 1
            continue

        filtered_indices, filtered_probs = filter_c100(topk_indices[sample_idx], topk_probs[sample_idx], is_ctc, skip, x)
        
        # --- 核心修改：在这里增加长度校验 ---
        if len(filtered_indices) != len(norm_true_text):
            stats['filtered_samples'] += 1
            if is_correct:
                stats['filtered_correct'] += 1
            continue
        # --- 结束修改 ---

        gt_c100 = generate_gt_c100(norm_true_text, ocr_chars, filtered_indices)
        
        if gt_c100 is None:
            stats['filtered_samples'] += 1
            if is_correct:
                stats['filtered_correct'] += 1
            continue

        stats['retained_samples'] += 1
        if is_correct:
            stats['retained_correct'] += 1

        lm_c100 = generate_lm_probs(norm_decoded_text, lm_model, ocr_to_lm, filtered_indices, lm_model.device)
        ocr_c100 = [(np.exp(probs) / np.sum(np.exp(probs))).tolist() for probs in filtered_probs][1:]
        lm_c100 = lm_c100
        gt_c100 = gt_c100[1:]
        
        if len(ocr_c100) != len(lm_c100) or len(ocr_c100) != len(gt_c100):
            stats['retained_samples'] -= 1
            if is_correct:
                stats['retained_correct'] -= 1
            continue

        local_train_data.append({
            'ocr_c100': ocr_c100,
            'lm_c100': lm_c100,
            'gt_c100': gt_c100
        })

    return local_train_data, stats

def generate_lm_probs(decoded_text, lm_model, ocr_to_lm, topk_indices, device):
    lm_c100_list = []
    for t in range(len(decoded_text)):
        prompt = decoded_text[:t] if t > 0 else ""
        probs = lm_model.get_full_probs(prompt)
        if probs is None:
            continue
        lm_probs_selected = []
        for ocr_idx in topk_indices[t]:
            lm_idx = ocr_to_lm.get(str(ocr_idx), -1)
            if lm_idx != -1:
                lm_probs_selected.append(probs[lm_idx])
            else:
                lm_probs_selected.append(0.0)
        lm_probs_selected = np.exp(lm_probs_selected) / np.sum(np.exp(lm_probs_selected))
        lm_c100_list.append(lm_probs_selected.tolist())
    return lm_c100_list

def process_json_folder(json_folder, map_file, ocr_char_file, lm_model_path, is_ctc, skip, output_file):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise SystemExit("错误：未找到任何CUDA GPU。此脚本需要至少一个GPU才能进行分布式处理。")
    print(f"检测到 {num_gpus} 个可用GPU，开始分布式处理...")

    json_files = [os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith('.json')]
    tasks = [(json_file, i % num_gpus, map_file, ocr_char_file, lm_model_path, is_ctc, skip) for i, json_file in enumerate(json_files)]

    all_train_data = []
    total_stats = {
        'total_samples': 0,
        'filtered_samples': 0,
        'filtered_correct': 0,
        'retained_samples': 0,
        'retained_correct': 0
    }
    
    with multiprocessing.get_context("spawn").Pool(processes=num_gpus) as pool:
        # 使用tqdm包装imap_unordered以显示进度条
        results_iterator = pool.imap_unordered(process_single_file_worker, tasks)
        
        # 创建tqdm进度条
        pbar = tqdm(results_iterator, total=len(tasks), desc="处理JSON文件")

        for result_from_worker in pbar:
            if result_from_worker:
                local_train_data, local_stats = result_from_worker
                all_train_data.extend(local_train_data)
                for key in total_stats:
                    total_stats[key] += local_stats[key]

    print("\n统计信息：")
    print(f"1. 样本的总数量: {total_stats['total_samples']}")
    print(f"2. 过滤时被筛掉的样本总数量: {total_stats['filtered_samples']}")
    print(f"   - 其中识别正确的数量: {total_stats['filtered_correct']}")
    print(f"3. 过滤时被保留的样本总数量: {total_stats['retained_samples']}")
    print(f"   - 其中识别正确的数量: {total_stats['retained_correct']}")

    print(f"\n处理完成。共聚合 {len(all_train_data)} 个有效样本。")
    print(f"正在保存到文件: {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_train_data, f, ensure_ascii=False)
    print("保存成功！")

if __name__ == "__main__":
    process_json_folder(
        json_folder='./ctc_probs/ctc_probs_svtr_train',
        map_file='map.json',
        ocr_char_file='ppocr_keys_v1.txt',
        lm_model_path='./checkpoints/char_transformer_epoch9.pt',
        is_ctc=True,
        skip=True,
        output_file='./output_train_long/svtr_train_long.json'
    )
    
# smtr lister :False
# crnn svtr   :True