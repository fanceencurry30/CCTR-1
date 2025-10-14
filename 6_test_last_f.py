import os
import json
import torch
import torch.nn.functional as F
import re
import numpy as np

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.utils.logging import get_logger
from openrec.modeling import build_model as build_rec_model
from tools.utils.ckpt import load_ckpt
from function import ProbabilityGenerator
from fusion_model_60p import CrossAttentionFusion
from openrec.postprocess import build_post_process
from rapidfuzz.distance import Levenshtein
import unicodedata

try:
    from opencc import OpenCC
    cc = OpenCC('t2s')  # Convert Traditional Chinese to Simplified Chinese
except:
    cc = None

def normalize_symbols(text):
    # 1. Convert full-width to half-width (including letters, numbers, punctuation)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Convert Traditional Chinese to Simplified Chinese
    if cc:
        text = cc.convert(text)
    
    # 3. Convert uppercase to lowercase
    text = text.lower()
    
    # 4. Remove all spaces (including half-width/full-width spaces)
    text = re.sub(r'\s+', '', text)
    
    # Custom symbol mapping
    text = re.sub(r'[【】]', lambda x: '[' if x.group(0) == '【' else ']', text)
    text = re.sub(r'[:：]', ':', text)
    text = re.sub(r'[，,]', ',', text)
    
    return text

def generate_lm_probs(decoded_text, lm_model, ocr_to_lm, topk_indices, device):
    """Generate top-100 probability distribution from LM model"""
    lm_c100_list = []
    for t in range(len(decoded_text)):
        prompt = decoded_text[:t] if t > 0 else ""
        probs = lm_model.get_full_probs(prompt)
        if probs is None:
            continue
        lm_probs_selected = []
        for ocr_idx in topk_indices[t]:
            lm_idx = ocr_to_lm.get(str(ocr_idx), -1)
            lm_probs_selected.append(probs[lm_idx] if lm_idx != -1 else 0.0)
        lm_probs_selected = np.exp(lm_probs_selected) / np.sum(np.exp(lm_probs_selected))
        lm_c100_list.append(lm_probs_selected.tolist())
    return lm_c100_list

def filter_and_extract_c100(preds, topk_indices, is_ctc, ocr_chars):
    """Filter and extract top-100 probability distribution and sum for each position"""
    filtered_indices = []
    filtered_sums = []
    prev_raw_idx = None
    x = len(ocr_chars)
    preds_probs = torch.softmax(preds, dim=-1).cpu().numpy()
    
    for t in range(preds.shape[0]):
        top1_idx = topk_indices[t][0]
        if is_ctc:
            if top1_idx == 0:  # CTC blank
                prev_raw_idx = top1_idx
                continue
            if prev_raw_idx is not None and top1_idx == prev_raw_idx:
                prev_raw_idx = top1_idx
                continue
            if not (1 <= top1_idx <= x):
                continue
            prev_raw_idx = top1_idx
        else:
            if top1_idx == 0:  # End symbol for non-CTC
                break
            if not (1 <= top1_idx <= x):
                continue
        topk_probs_sum = preds_probs[t, topk_indices[t]].sum()
        filtered_indices.append(topk_indices[t])
        filtered_sums.append(topk_probs_sum)
    return filtered_indices, filtered_sums, preds_probs

def generate_gt_c100(true_text, ocr_chars, topk_indices):
    """Generate Ground Truth top-100 probability distribution"""
    gt_c100_list = []
    if len(true_text) != len(topk_indices):
        return None
    for t, char in enumerate(true_text):
        if char not in ocr_chars:
            return None
        ocr_idx = ocr_chars.index(char) + 1
        if ocr_idx not in topk_indices[t]:
            return None
        gt_probs = [1.0 if idx == ocr_idx else 0.0 for idx in topk_indices[t]]
        gt_c100_list.append(gt_probs)
    return gt_c100_list

def fuse_and_replace(preds, filtered_indices, filtered_sums, ocr_c100, lm_c100, fusion_model, device, threshold, gt_c100):
    """Fuse and put probabilities back into the original CTC sequence"""
    preds_probs = torch.softmax(preds, dim=-1).cpu().numpy()  # [seq_len, vocab_size]
    fused_preds = preds_probs.copy()  # Keep a copy of the original predictions
    fused_c100 = []
    
    if threshold == 0:
        return preds, ocr_c100  # Directly return original preds, avoid conversion
    
    for t in range(1, len(filtered_indices)):  # Start from t=1
        ocr_prob = torch.tensor(ocr_c100[t-1], device=device).unsqueeze(0)  # [1, 100]
        top1_prob = ocr_prob.max().item()
        is_negative = np.argmax(ocr_c100[t-1]) != np.argmax(gt_c100[t-1])
        if top1_prob < threshold and is_negative:  # Only fuse for negative samples and low-confidence characters
            lm_prob = torch.tensor(lm_c100[t-1], device=device).unsqueeze(0)  # [1, 100]
            with torch.no_grad():
                fused_prob = fusion_model(ocr_prob, lm_prob).cpu().numpy()[0]  # [100]
                fused_probs_scaled = fused_prob * filtered_sums[t]
                fused_preds[t, filtered_indices[t]] = fused_probs_scaled
                fused_c100.append(fused_prob.tolist())
        else:
            fused_c100.append(ocr_c100[t-1])  # Keep original probability for positive samples or high confidence
    
    return torch.from_numpy(fused_preds).float().to(device), fused_c100

def extract_decoded_text(decoded_item):
    if isinstance(decoded_item, tuple) and len(decoded_item) >= 1:
        return decoded_item[0]
    elif isinstance(decoded_item, str):
        return decoded_item
    else:
        raise ValueError(f"Unexpected decoded item type: {type(decoded_item)}")

def main(config_path, model_path, lm_model_path, fusion_model_path, residual_model_path, output_dir, model_type='smtr', is_training=False, trsd=0.01):
    # Load config file
    cfg = Config(config_path)
    cfg = cfg.cfg

    # Set up logger
    logger = get_logger('recognition_fusion', os.path.join(output_dir, 'recognition_fusion.log'))
    logger.info(f"Loaded config file: {config_path}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build post-process module
    post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])
    logger.info("Post-process module built")

    # Build model
    char_num = post_process_class.get_character_num()
    cfg['Architecture']['Decoder']['out_channels'] = char_num
    model = build_rec_model(cfg['Architecture'])
    load_ckpt(model, cfg, None, None)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded: {model_path}")

    # Load LM model and fusion model
    lm_model = ProbabilityGenerator(lm_model_path)
    fusion_model = CrossAttentionFusion().to(device)
    fusion_model.load_state_dict(torch.load(fusion_model_path, map_location=device))
    fusion_model.eval()
    logger.info(f"Fusion model loaded: {fusion_model_path}")

    # Load mapping and character table
    with open('map.json', 'r', encoding='utf-8') as f:
        ocr_to_lm = json.load(f)
    with open('ppocr_keys_v1.txt', 'r', encoding='utf-8') as f:
        ocr_chars = [line.strip() for line in f.readlines()]

    # Build dataloader
    cfg['Global']['distributed'] = False
    eval_dataloader = build_dataloader(cfg, 'Eval', logger, task='rec')
    logger.info(f"Evaluation dataloader built, contains {len(eval_dataloader)} batches")

    # Define threshold
    threshold = trsd

    # Statistics
    stats = {
        'total_samples': 0,
        'correct_total_before': 0,
        'correct_total_after': 0,
        'filtered_samples': 0,
        'filtered_correct_before': 0,
        'filtered_correct_after': 0,
        # CER statistics
        'char_errors_total_before': 0,
        'char_errors_total_after': 0,
        'gt_chars_total_before': 0,
        'gt_chars_total_after': 0,
        'cer_before_list': [],
        'cer_after_list': [],
        'cer_before_list_filtered': [],
        'cer_after_list_filtered': [],
        'filtered_char_errors_before': 0,
        'filtered_gt_chars_before': 0,
        'filtered_char_errors_after': 0,
        'filtered_gt_chars_after': 0,
        # Negative sample statistics
        'negative_chars': 0,
        'corrected_chars': 0
    }

    # Initialize training data list
    train_data = []

    for batch_idx, batch in enumerate(eval_dataloader):
        batch_tensor = [t.to(device) for t in batch]

        with torch.no_grad():
            preds = model(batch_tensor[0], data=batch_tensor[1:])

            # Adjust preds format to fit post-process
            if model_type.lower() == 'ctc':
                if isinstance(preds, torch.Tensor):
                    preds = preds.cpu().numpy()
                if not is_training:
                    preds = F.softmax(torch.from_numpy(preds), dim=-1).numpy()
                preds_for_postprocess = preds
                batch_numpy = [t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in batch_tensor]
            elif model_type.lower() == 'lister':
                if isinstance(preds, list) and len(preds) == 2 and isinstance(preds[1], dict) and 'logits' in preds[1]:
                    logits = preds[1]['logits']
                    if not is_training:
                        logits = F.softmax(logits, dim=-1)
                    preds_for_postprocess = [None, {'logits': logits}]
                else:
                    raise ValueError("Abnormal LISTER model output format")
                batch_numpy = batch_tensor
            elif model_type.lower() == 'smtr':
                if is_training and isinstance(preds, list) and len(preds) == 2:
                    preds_for_postprocess = preds[1]
                else:
                    preds_for_postprocess = preds
                batch_numpy = batch_tensor
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Decode predictions before fusion (whole batch)
            if model_type.lower() == 'ctc':
                decoded_output_before = post_process_class(preds_for_postprocess, batch_numpy, torch_tensor=False)
            else:
                decoded_output_before = post_process_class(preds_for_postprocess, batch_tensor, torch_tensor=False)
            
            if isinstance(decoded_output_before, tuple) and len(decoded_output_before) == 2:
                decoded_texts_before, labels = decoded_output_before
            elif isinstance(decoded_output_before, list):
                decoded_texts_before = decoded_output_before
                labels = [None] * len(decoded_texts_before)
            else:
                raise ValueError(f"Abnormal return type from post_process_class: {type(decoded_output_before)}")

            # Process each sample
            for sample_idx in range(len(decoded_texts_before)):
                stats['total_samples'] += 1
                # Extract ground truth label text
                true_text = ""
                if labels and labels[sample_idx] is not None:
                    if isinstance(labels[sample_idx], tuple):
                        true_text = labels[sample_idx][0]
                    elif isinstance(labels[sample_idx], str):
                        true_text = labels[sample_idx]
                
                # Extract decoded text before fusion
                decoded_text_before = extract_decoded_text(decoded_texts_before[sample_idx])

                # Normalize text and calculate accuracy before fusion
                norm_true_text = normalize_symbols(true_text)
                norm_decoded_text_before = normalize_symbols(decoded_text_before)
                is_correct_before = norm_true_text == norm_decoded_text_before
                if is_correct_before:
                    stats['correct_total_before'] += 1

                # Calculate CER before fusion (global statistics)
                char_errors_before = Levenshtein.distance(norm_decoded_text_before, norm_true_text)
                stats['char_errors_total_before'] += char_errors_before
                stats['gt_chars_total_before'] += len(norm_true_text)
                cer_before = char_errors_before / (len(norm_true_text) + 1e-5)
                stats['cer_before_list'].append(cer_before)

                true_len = len(true_text)
                decoded_len = len(decoded_text_before)

                # Check for empty strings and skip
                if true_len == 0 or decoded_len == 0:
                    stats['char_errors_total_after'] += char_errors_before
                    stats['gt_chars_total_after'] += len(norm_true_text) if len(norm_true_text) > 0 else 1
                    stats['cer_after_list'].append(cer_before)
                    if is_correct_before:
                        stats['correct_total_after'] += 1
                    continue

                # Filtering condition
                if true_len == 1:
                    stats['char_errors_total_after'] += char_errors_before
                    stats['gt_chars_total_after'] += len(norm_true_text) if len(norm_true_text) > 0 else 1
                    stats['cer_after_list'].append(cer_before)
                    if is_correct_before:
                        stats['correct_total_after'] += 1
                    continue
                
                if true_len != decoded_len:
                    stats['char_errors_total_after'] += char_errors_before
                    stats['gt_chars_total_after'] += len(norm_true_text) if len(norm_true_text) > 0 else 1
                    stats['cer_after_list'].append(cer_before)
                    if is_correct_before:
                        stats['correct_total_after'] += 1
                    continue

                # Get prediction probabilities for a single sample
                if model_type.lower() == 'lister':
                    smtr_probs = preds_for_postprocess[1]['logits'][sample_idx]
                else:
                    smtr_probs = preds_for_postprocess[sample_idx]
                    if isinstance(smtr_probs, np.ndarray):
                        smtr_probs = torch.from_numpy(smtr_probs).to(device)
                topk_probs, topk_indices = torch.topk(smtr_probs, k=100, dim=-1, largest=True, sorted=True)

                # Filter and extract C100 and sum
                filtered_indices, filtered_sums, preds_probs = filter_and_extract_c100(
                    smtr_probs, topk_indices.cpu().numpy(), model_type.lower() == 'ctc', ocr_chars
                )
                if not filtered_indices:
                    stats['char_errors_total_after'] += char_errors_before
                    stats['gt_chars_total_after'] += len(norm_true_text) if len(norm_true_text) > 0 else 1
                    stats['cer_after_list'].append(cer_before)
                    if is_correct_before:
                        stats['correct_total_after'] += 1
                    continue

                # Generate GT C100
                gt_c100 = generate_gt_c100(true_text, ocr_chars, filtered_indices)
                if gt_c100 is None:
                    stats['char_errors_total_after'] += char_errors_before
                    stats['gt_chars_total_after'] += len(norm_true_text) if len(norm_true_text) > 0 else 1
                    stats['cer_after_list'].append(cer_before)
                    if is_correct_before:
                        stats['correct_total_after'] += 1
                    continue

                # Generate OCR C100 and LM C100
                ocr_c100 = [(preds_probs[t, filtered_indices[t]] / filtered_sums[t]).tolist() for t in range(len(filtered_indices))][1:]
                lm_c100 = generate_lm_probs(decoded_text_before, lm_model, ocr_to_lm, filtered_indices, device)
                gt_c100 = gt_c100[1:]

                if len(ocr_c100) != len(lm_c100) or len(ocr_c100) != len(gt_c100):
                    stats['char_errors_total_after'] += char_errors_before
                    stats['gt_chars_total_after'] += len(norm_true_text) if len(norm_true_text) > 0 else 1
                    stats['cer_after_list'].append(cer_before)
                    if is_correct_before:
                        stats['correct_total_after'] += 1
                    continue

                stats['filtered_samples'] += 1
                if is_correct_before:
                    stats['filtered_correct_before'] += 1

                # Collect filtered sample data
                train_data.append({
                    "ocr_c100": ocr_c100,
                    "lm_c100": lm_c100,
                    "gt_c100": gt_c100
                })

                # Fuse and put back
                fused_preds_sample, fused_c100 = fuse_and_replace(
                    smtr_probs, filtered_indices, filtered_sums, ocr_c100, lm_c100, fusion_model, device, threshold, gt_c100
                )

                # Build fused preds_for_postprocess (per sample)
                if model_type.lower() == 'ctc':
                    fused_preds_for_postprocess = fused_preds_sample.unsqueeze(0).cpu().numpy()
                    single_batch_numpy = [t[sample_idx:sample_idx+1].cpu().numpy() for t in batch_tensor]
                    decoded_output_after = post_process_class(fused_preds_for_postprocess, single_batch_numpy, torch_tensor=False)
                elif model_type.lower() == 'lister':
                    fused_preds_for_postprocess = [None, {'logits': fused_preds_sample.unsqueeze(0)}]
                    single_batch_tensor = [t[sample_idx:sample_idx+1].to(device) for t in batch_tensor]
                    decoded_output_after = post_process_class(fused_preds_for_postprocess, single_batch_tensor, torch_tensor=False)
                elif model_type.lower() == 'smtr':
                    fused_preds_for_postprocess = fused_preds_sample.unsqueeze(0)
                    single_batch_tensor = [t[sample_idx:sample_idx+1].to(device) for t in batch_tensor]
                    decoded_output_after = post_process_class(fused_preds_for_postprocess, single_batch_tensor, torch_tensor=False)
                
                if isinstance(decoded_output_after, tuple) and len(decoded_output_after) == 2:
                    decoded_texts_after, _ = decoded_output_after
                elif isinstance(decoded_output_after, list):
                    decoded_texts_after = decoded_output_after
                else:
                    raise ValueError(f"Abnormal return type from post_process_class: {type(decoded_output_after)}")

                # Extract decoded text after fusion
                decoded_text_after = extract_decoded_text(decoded_texts_after[0])

                # Calculate accuracy after fusion
                norm_decoded_text_after = normalize_symbols(decoded_text_after)
                is_correct_after = norm_true_text == norm_decoded_text_after
                if is_correct_after:
                    stats['correct_total_after'] += 1
                if is_correct_after and stats['filtered_samples'] > 0:
                    stats['filtered_correct_after'] += 1
                
                 # Count number of negative sample characters and corrections
                for t in range(len(ocr_c100)):
                    if np.argmax(ocr_c100[t]) != np.argmax(gt_c100[t]):
                        stats['negative_chars'] += 1
                        if np.argmax(fused_c100[t]) == np.argmax(gt_c100[t]):
                            stats['corrected_chars'] += 1
                
                # Calculate CER after fusion (global statistics)
                char_errors_after = Levenshtein.distance(norm_decoded_text_after, norm_true_text)
                stats['char_errors_total_after'] += char_errors_after
                stats['gt_chars_total_after'] += len(norm_true_text)
                cer_after = char_errors_after / (len(norm_true_text) + 1e-5)
                stats['cer_after_list'].append(cer_after)
                # For filtered samples, add to filtered CER
                stats['cer_before_list_filtered'].append(cer_before)
                stats['filtered_char_errors_before'] += char_errors_before
                stats['filtered_gt_chars_before'] += len(norm_true_text)
                stats['cer_after_list_filtered'].append(cer_after)
                stats['filtered_char_errors_after'] += char_errors_after
                stats['filtered_gt_chars_after'] += len(norm_true_text)

    # Calculate correction rate
    correction_rate = stats['corrected_chars'] / stats['negative_chars'] if stats['negative_chars'] > 0 else 0.0

    # Output statistics
    # print("\nStatistics:")
    # print("### Before filtering:")
    # print(f"- Total samples: {stats['total_samples']}")
    # print(f"- Correct recognitions before fusion: {stats['correct_total_before']} (Accuracy: {stats['correct_total_before']/stats['total_samples']:.4f})")
    # print(f"- Correct recognitions after fusion: {stats['correct_total_after']} (Accuracy: {stats['correct_total_after']/stats['total_samples']:.4f})")
    # print(f"- Mean CER before fusion (before filtering): {np.mean(stats['cer_before_list'])*100:.2f}%")
    # print(f"- Mean CER after fusion (before filtering): {np.mean(stats['cer_after_list'])*100:.2f}%")
    # print(f"- Global CER before fusion (before filtering): {stats['char_errors_total_before'] / (stats['gt_chars_total_before'] + 1e-5):.4f}")
    # print(f"- Global CER after fusion (before filtering): {stats['char_errors_total_after'] / (stats['gt_chars_total_after'] + 1e-5):.4f}")
    # print("### After filtering:")
    # print(f"- Total filtered samples: {stats['filtered_samples']}")
    # print(f"- Correct recognitions before fusion (filtered): {stats['filtered_correct_before']} (Accuracy: {stats['filtered_correct_before']/stats['filtered_samples'] if stats['filtered_samples'] > 0 else 0:.4f})")
    # print(f"- Correct recognitions after fusion (filtered): {stats['filtered_correct_after']} (Accuracy: {stats['filtered_correct_after']/stats['filtered_samples'] if stats['filtered_samples'] > 0 else 0:.4f})")
    # print(f"- Mean CER before fusion (filtered): {np.mean(stats['cer_before_list_filtered'])*100 if stats['filtered_samples'] > 0 else 0:.2f}%")
    # print(f"- Mean CER after fusion (filtered): {np.mean(stats['cer_after_list_filtered'])*100 if stats['filtered_samples'] > 0 else 0:.2f}%")
    # print(f"- Global CER before fusion (filtered): {stats['filtered_char_errors_before'] / (stats['filtered_gt_chars_before'] + 1e-5):.4f}")
    # print(f"- Global CER after fusion (filtered): {stats['filtered_char_errors_after'] / (stats['filtered_gt_chars_after'] + 1e-5):.4f}")
    print("### Negative samples:")
    print(f"- Number of negative sample characters after filtering: {stats['negative_chars']}")
    print(f"- Number of corrected to positive sample characters after filtering: {stats['corrected_chars']}")
    print(f"- Correction rate: {correction_rate:.4f}")

    # Save training data to JSON file
    with open(os.path.join(output_dir, 'train_data.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)

if __name__ == "__main__":
    config_path = './method_yml/svtr.yml'
    model_path = './method_pth/best_svtr.pth'
    lm_model_path = './checkpoints/char_transformer_epoch8.pt'
    fusion_model_path = './models_60p/svtr_final/fusion_model_best_fusion.pth'
    residual_model_path = './models_60p/svtr_final/fusion_model_best_residual_ocrlm_lister_all.pth'
    output_dir = './output_test/svtr/art_long/'
    model_type = 'ctc'
    os.makedirs(output_dir, exist_ok=True)
    trs = 1.0
    main(config_path, model_path, lm_model_path, fusion_model_path, residual_model_path, output_dir, model_type=model_type, trsd=trs)