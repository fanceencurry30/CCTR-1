import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from openrec.modeling import build_model as build_rec_model
from openrec.postprocess import build_post_process
from tools.engine.config import Config
from tools.data import build_dataloader
from tools.utils.ckpt import load_ckpt
from tools.utils.logging import get_logger

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

# 生成类似 CTC 的 top-100 概率和解码文本的函数
def generate_lister_ctc_data(config_path, model_path, output_dir):
    # 加载配置文件
    cfg = Config(config_path)
    cfg = cfg.cfg

    # 设置日志
    logger = get_logger('generate_lister_ctc_data', os.path.join(output_dir, 'generate_lister_ctc_data.log'))
    logger.info(f"加载配置文件: {config_path}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 构建后处理模块
    post_process_class = build_post_process(cfg['PostProcess'], cfg['Global'])
    logger.info("后处理模块已构建")

    # 构建模型
    char_num = post_process_class.get_character_num()
    # 为复杂解码器（如 LISTERDecoder）注入 out_channels
    if 'lister_decoder' in cfg['Architecture']['Decoder']:
        cfg['Architecture']['Decoder']['lister_decoder']['out_channels'] = char_num
    # 为简单解码器设置 out_channels
    cfg['Architecture']['Decoder']['out_channels'] = char_num

    model = build_rec_model(cfg['Architecture'])
    load_ckpt(model, cfg, None, None)
    model.to(device)
    model.eval()
    logger.info(f"模型已加载: {model_path}")

    # 构建数据加载器
    cfg['Global']['distributed'] = False
    train_dataloader = build_dataloader(cfg, 'Eval', logger, task='rec')
    logger.info(f"评估数据加载器已构建，包含 {len(train_dataloader)} 个批次")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # 生成类似 CTC 的 top-100 概率和解码文本
    for batch_idx, batch in enumerate(train_dataloader):
        batch_tensor = [t.to(device) for t in batch]
        batch_numpy = [t.cpu().numpy() for t in batch]

        with torch.no_grad():
            preds = model(batch_tensor[0], data=batch_tensor[1:])

            # 根据模型输出的类型来正确提取 logits
            # 情况1: postprocess=True. preds 是一个元组 (decoded_text, other_info)
            if isinstance(preds, tuple) and len(preds) > 1 and isinstance(
                    preds[1], dict) and 'logits' in preds[1]:
                # other_info['logits'] 是一个张量列表, 取最后一个
                ctc_logits = preds[1]['logits'][-1]
            # 情况2: postprocess=False. preds 是一个字典列表 [{'logits': ...}, ...]
            elif isinstance(preds, list) and preds and isinstance(
                    preds[-1], dict) and 'logits' in preds[-1]:
                # 取列表里最后一个字典中的 'logits'
                ctc_logits = preds[-1]['logits']
            else:
                # 兼容其他模型或未预期的输出格式（例如直接输出tensor）
                ctc_logits = preds

            ctc_probs = torch.softmax(ctc_logits, dim=-1)#把logits转换为概率分布

            # 获取 top-100 概率及其索引
            topk_probs, topk_indices = torch.topk(ctc_probs, k=100, dim=-1, largest=True, sorted=True)

            # 使用后处理模块解码 成真实文本及对应的真实标签
            decoded_texts, labels = post_process_class(preds, batch_numpy)

            # 转换为 JSON 可序列化格式
            topk_probs_list = topk_probs.cpu().numpy().tolist()
            topk_indices_list = topk_indices.cpu().numpy().tolist()
            labels_list = list(labels)
            decoded_texts_list = [text[0] if isinstance(text, tuple) else text for text in decoded_texts]

            # 保存到 JSON 文件
            output_data = {
                'batch_idx': batch_idx,
                'topk_probs': topk_probs_list,      # [batch_size, seq_len, 100]
                'topk_indices': topk_indices_list,  # [batch_size, seq_len, 100]
                'labels': labels_list,              # 真实标签
                'decoded_texts': decoded_texts_list # LISTER 解码后的文本
            }
            output_path = os.path.join(output_dir, f'lister_ctc_probs_{batch_idx}.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False)

        logger.info(f"批次 {batch_idx} 的 top-100 类似 CTC 概率和解码文本已保存至 {output_path}")

if __name__ == "__main__":
    config_path = './method_yml/svtr.yml'
    model_path = './method_pth/best_svtr.pth'
    output_dir = './ctc_probs/ctc_probs_svtr_train'
    generate_lister_ctc_data(config_path, model_path, output_dir)
    print(f"Top-100 类似 CTC 概率和解码文本已生成并保存至 {output_dir}")