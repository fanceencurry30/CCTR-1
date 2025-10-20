# -*- coding: utf-8 -*-
"""
提供生成文本的实用函数
"""
import torch
import torch.nn.functional as F  # 导入F
from config import Config

def generate_text(model, char2id, prompt, max_length=50, temperature=1.0):
    """
    生成文本
    :param model: 训练好的模型
    :param char2id: 字符到ID的字典
    :param prompt: 起始文本（如"中国的首都是"）
    :param max_length: 生成的最大长度
    :param temperature: 温度参数（控制随机性）
    """
    model.eval()
    id2char = {v: k for k, v in char2id.items()}
    input_ids = [char2id.get(c, char2id['<UNK>']) for c in prompt]
    input_ids = torch.tensor([input_ids]).to(Config.device)
    
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)
        next_token_logits = logits[0, -1, :] / temperature
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    generated = ''.join([id2char.get(id.item(), '<UNK>') for id in input_ids[0]])
    return generated