import os
import json
import torch
from collections import defaultdict
import numpy as np
from openrec.modeling import build_model as build_rec_model
from openrec.postprocess import build_post_process
from tools.engine.config import Config
from tools.data import build_dataloader
from tools.utils.ckpt import load_ckpt
from tools.utils.logging import get_logger


def parse_label(label_str):
    """解析带元数据的label字符串，例如 'hello <image_id=1_line_id=2>'"""
    if '<image_id=' in label_str and '_line_id=' in label_str:
        parts = label_str.rsplit('<', 1)
        text = parts[0].strip()
        meta_str = '<' + parts[1]
        image_id = int(meta_str.split('image_id=')[1].split('_')[0])
        line_id = int(meta_str.split('line_id=')[1].split('>')[0])
        return text, image_id, line_id
    return label_str, -1, -1


def generate_lister_ctc_data(config_path, model_path, output_dir):
    # 加载配置
    cfg = Config(config_path).cfg
    logger = get_logger('generate_lister_ctc_data', os.path.join(output_dir, 'generate_lister_ctc_data.log'))

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])
    char_num = post_process_class.get_character_num()

    # 设置输出通道数
    if 'lister_decoder' in cfg['Architecture']['Decoder']:
        cfg['Architecture']['Decoder']['lister_decoder']['out_channels'] = char_num
    cfg['Architecture']['Decoder']['out_channels'] = char_num

    model = build_rec_model(cfg['Architecture'])
    load_ckpt(model, cfg, None, None)
    model.to(device)
    model.eval()

    # 构建数据加载器
    cfg['Global']['distributed'] = False
    dataloader = build_dataloader(cfg, 'Eval', logger, task='rec')

    os.makedirs(output_dir, exist_ok=True)

    # 主聚合结构
    image_results = defaultdict(lambda: {
        'lines': [],  # 保存每一行的完整结构
    })

    # 遍历 batch
    for batch_idx, batch in enumerate(dataloader):
        batch_tensor = [t.to(device) for t in batch]
        batch_numpy = [t.cpu().numpy() for t in batch]

        with torch.no_grad():
            preds = model(batch_tensor[0], data=batch_tensor[1:])

            # 取 logits
            if isinstance(preds, tuple) and len(preds) > 1 and isinstance(preds[1], dict) and 'logits' in preds[1]:
                ctc_logits = preds[1]['logits'][-1]
            elif isinstance(preds, list) and preds and isinstance(preds[-1], dict) and 'logits' in preds[-1]:
                ctc_logits = preds[-1]['logits']
            else:
                ctc_logits = preds

            ctc_probs = torch.softmax(ctc_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(ctc_probs, k=100, dim=-1, largest=True, sorted=True)

            # 解码文本
            decoded_texts, labels = post_process_class(preds, batch_numpy)
            decoded_texts = [text[0] if isinstance(text, tuple) else text for text in decoded_texts]

            for i in range(len(labels)):
                label_text = labels[i][0] if isinstance(labels[i], (list, tuple)) else labels[i]
                text, image_id, line_id = parse_label(label_text)
                if image_id == -1:
                    continue

                line_data = {
                    'line_id': line_id,
                    'decoded_text': decoded_texts[i],
                    'label': text,
                    'topk_probs': topk_probs[i].cpu().numpy().tolist(),
                    'topk_indices': topk_indices[i].cpu().numpy().tolist(),
                    'batch_idx': batch_idx
                }
                image_results[image_id]['lines'].append(line_data)

    # 整理输出结果
    final_results = []
    for img_id, data in image_results.items():
        # 按行号和 batch 顺序排序
        lines_sorted = sorted(data['lines'], key=lambda x: (x['line_id'], x['batch_idx']))

        # 拼接完整文本
        merged_decoded_text = "".join([line['decoded_text'] for line in lines_sorted])
        merged_label = "".join([line['label'] for line in lines_sorted])

        merged_probs = np.concatenate([np.array(line['topk_probs']) for line in lines_sorted], axis=0)
        merged_indices = np.concatenate([np.array(line['topk_indices']) for line in lines_sorted], axis=0)

        result = {
            'image_id': img_id,
            'decoded_text': merged_decoded_text,
            'label': merged_label,
            'num_lines': len(lines_sorted),
            'topk_probs': merged_probs.tolist(),
            'topk_indices': merged_indices.tolist(),
            'line_ids': [line['line_id'] for line in lines_sorted]
        }
        final_results.append(result)

        # 保存单图结果
        output_path = os.path.join(output_dir, f'image_{img_id}_result.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"图片 {img_id} 结果已保存 ({result['num_lines']} 行文本)")

    # 汇总信息
    final_results.sort(key=lambda x: x['image_id'])
    summary = {
        'total_images': len(final_results),
        'total_lines': sum(r['num_lines'] for r in final_results),
        'output_dir': os.path.abspath(output_dir)
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"处理完成！共处理 {summary['total_images']} 张图片，{summary['total_lines']} 行文本")
    return final_results


if __name__ == "__main__":
    config_path = './method_yml/lister.yml'
    model_path = './method_pth/best_lister.pth'
    output_dir = './ctc_probs/ctc_probs_svtr_train_new_100_lister_final'
    generate_lister_ctc_data(config_path, model_path, output_dir)
    print(f"✅ Top-100 CTC 概率和解码文本已生成并保存至 {output_dir}")