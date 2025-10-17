# -*- coding: utf-8 -*-
"""
返回下一个字符Top-k概率分布的可调用函数
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
        model.eval()  # 关键：保持模型推理模式
        return char2id, model

    def get_topk_probabilities(self, prompt, topk=5):
        """
        获取下一个字符的Top-k概率分布（纯数据返回）
        Args:
            prompt (str): 输入文本
            topk (int): 需要返回的Top数量（默认5）
        Returns:
            list: Top-k的字符概率列表（元素为元组：(字符, 概率)）
            int: 输入中有效字符的数量（在词表中的字符数）
        """
        # 过滤无效字符（不在词表中的字符）
        valid_chars = [c for c in prompt if c in self.char2id]
        if not valid_chars:
            return [], 0  # 无有效字符时返回空列表和0
        
        # 转换输入为模型需要的张量
        input_ids = [self.char2id[c] for c in valid_chars]
        input_tensor = torch.tensor([input_ids], device=self.device)

        # 模型推理（无梯度模式加速）
        with torch.no_grad():
            logits = self.model(input_tensor)[0, -1, :]  # 取最后一个字符的输出
            probs = torch.softmax(logits, dim=-1)  # 转换为概率分布
        
        # 获取Top-k的字符和概率（按概率降序排列）
        topk_probs, topk_ids = torch.topk(probs, topk)
        results = [
            (self.id2char[idx.item()], prob.item())  # 转换为（字符，概率）元组
            for idx, prob in zip(topk_ids, topk_probs)
        ]

        return results, len(valid_chars)

# 使用示例（仅演示调用方式，无实际输出）
if __name__ == "__main__":
    # 初始化生成器（加载模型和词表）
    generator = ProbabilityGenerator(
        checkpoint_path="/home/u2024000980/sanliwan/chartransformer_1/char_transformer_epoch9.pt"
    )

    # 输入文本
    input_text = "在務基本上就依托在農業社的基"

    # 调用函数获取Top-5概率（无打印，仅返回数据）
    top5_results, valid_len = generator.get_topk_probabilities(input_text, topk=5)

#调用示例
# from function import ProbabilityGenerator 

# def main():
#     # ------------------------- 步骤1：初始化生成器 -------------------------
#     # 模型检查点路径（替换为你的实际路径）
#     checkpoint_path = "/home/u2024000980/sanliwan/chartransformer_1/char_transformer_epoch4.pt"
#     generator = ProbabilityGenerator(checkpoint_path)

#     # ------------------------- 步骤2：调用预测函数 -------------------------
#     # 输入文本（替换为你的实际输入）
#     input_text = "在務基本上就依托在農業社的基"
#     # 调用函数获取 Top-5 概率分布（topk=5 为默认值，可省略）
#     top5_results, valid_len = generator.get_topk_probabilities(input_text, topk=5)

#     # ------------------------- 步骤3：处理结果 -------------------------
#     if top5_results:  # 输入有效（存在至少一个有效字符）
#         print(f"输入文本有效字符数: {valid_len}")
#         print("Top-5 字符概率分布:")
#         for char, prob in top5_results:
#             print(f"  字符：{char}，概率：{prob:.6f}")
#     else:
#         print("输入无效：文本中无词表包含的字符")

# if __name__ == "__main__":
#     main()

