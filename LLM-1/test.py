# -*- coding: utf-8 -*-
"""
生成下一个字符的 Top-5 概率预测
"""
import json
import torch
from model import CharTransformer
from config import Config

class ProbabilityGenerator:
    def __init__(self, checkpoint_path):
        # 初始化模型和字典
        self.device = Config.device
        self.char2id, self.model = self._load_resources(checkpoint_path)
        self.id2char = {v: k for k, v in self.char2id.items()}
    
    def _load_resources(self, checkpoint_path):
        """加载模型和字典"""
        with open(Config.vocab_path, 'r', encoding='utf-8') as f:
            char2id = json.load(f)
        
        model = CharTransformer(len(char2id)).to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        return char2id, model

    def predict_next_char(self, prompt, topk=5):
        """
        预测下一个字符，并输出 top-k 的候选及概率
        """
        valid_chars = [c for c in prompt if c in self.char2id]
        if not valid_chars:
            return None, 0

        input_ids = [self.char2id[c] for c in valid_chars]
        input_tensor = torch.tensor([input_ids], device=self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)[0, -1, :]  # 取最后一个字符的输出
            probs = torch.softmax(logits, dim=-1)

        # top-k
        topk_probs, topk_ids = torch.topk(probs, topk)
        results = [(self.id2char[idx.item()], prob.item()) for idx, prob in zip(topk_ids, topk_probs)]

        return results, len(valid_chars)

# 使用示例
if __name__ == "__main__":
    generator = ProbabilityGenerator(
        checkpoint_path="/home/u2024000980/sanliwan/chartransformer_1/char5.6/char_transformer_epoch9.pt"
    )

    input_text = "在務基本上就依托在農業社的基"
    results, valid_len = generator.predict_next_char(input_text, topk=5)

    if results:
        print(f"输入文本: {input_text}")
        print(f"模型预测下一个字符的 Top 5 结果：")
        for char, prob in results:
            print(f"  字符：{char}，概率：{prob:.4f}")
    else:
        print("输入无效，无有效字符")
