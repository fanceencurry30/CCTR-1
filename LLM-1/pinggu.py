# -*- coding: utf-8 -*-
"""
模型性能评估脚本
"""
import json
import torch
import numpy as np
from tqdm import tqdm
from model import CharTransformer
from config import Config
from torch.utils.data import DataLoader
from data_loader import CharDataset, collate_fn
import torch.nn.functional as F


class ModelEvaluator:
    def __init__(self, checkpoint_path, vocab_path, test_data_path):
        self.device = Config.device
        self.char2id = self._load_vocab(vocab_path)
        self.model = self._load_model(checkpoint_path, len(self.char2id))
        self.test_loader = self._prepare_dataloader(test_data_path)
        
    def _load_vocab(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_model(self, checkpoint_path, vocab_size):
        model = CharTransformer(vocab_size).to(self.device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        model.eval()
        return model
    
    def _prepare_dataloader(self, data_path):
        dataset = CharDataset(data_path, self.char2id, Config.max_seq_len)
        return DataLoader(
            dataset,
            batch_size=64,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    
    def evaluate(self):
        """执行完整评估流程"""
        metrics = {
            'total_samples': 0,
            'total_tokens': 0,
            'perplexity': 0.0,
            'avg_rank_percent': 0.0,
            'top1_acc': 0.0,
            'top5_acc': 0.0,
            'top10_acc': 0.0,
            'avg_rank': 0.0
        }
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 模型前向传播
                logits = self.model(inputs)
                
                # 计算各指标
                batch_metrics = self._calculate_metrics(logits, targets)
                
                # 累加指标
                metrics['total_samples'] += batch_metrics['batch_size']
                metrics['total_tokens'] += batch_metrics['num_tokens']
                metrics['perplexity'] += batch_metrics['perplexity']
                metrics['avg_rank_percent'] += batch_metrics['rank_percent_sum']
                metrics['top1_acc'] += batch_metrics['top1_correct']
                metrics['top5_acc'] += batch_metrics['top5_correct']
                metrics['top10_acc'] += batch_metrics['top10_correct']
                metrics['avg_rank'] += batch_metrics['rank_sum']
        
        # 计算最终指标
        return self._finalize_metrics(metrics)
    
    def _calculate_metrics(self, logits, targets):
        """计算单个batch的指标"""
        batch_size, seq_len = targets.size()
        vocab_size = logits.size(-1)
        
        # 计算交叉熵损失（用于困惑度）
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            reduction='none'
        )
        perplexity = torch.exp(loss.mean())
        
        # 获取预测概率
        probs = torch.softmax(logits, dim=-1)
        
        # 获取目标字符的索引
        target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        
        # 计算排名相关指标
        sorted_indices = torch.argsort(probs, dim=-1, descending=True)
        ranks = (sorted_indices == targets.unsqueeze(-1)).nonzero()[:, -1] + 1
        
        return {
            'batch_size': batch_size,
            'num_tokens': batch_size * seq_len,
            'perplexity': perplexity.item() * batch_size * seq_len,
            'rank_percent_sum': (ranks.float() / vocab_size).sum().item(),
            'rank_sum': ranks.sum().item(),
            'top1_correct': (ranks == 1).sum().item(),
            'top5_correct': (ranks <= 5).sum().item(),
            'top10_correct': (ranks <= 10).sum().item()
        }
    
    def _finalize_metrics(self, metrics):
        """汇总最终指标"""
        return {
            'perplexity': metrics['perplexity'] / metrics['total_tokens'],
            'avg_rank_percent': metrics['avg_rank_percent'] / metrics['total_tokens'],
            'avg_rank': metrics['avg_rank'] / metrics['total_tokens'],
            'top1_accuracy': metrics['top1_acc'] / metrics['total_tokens'],
            'top5_accuracy': metrics['top5_acc'] / metrics['total_tokens'],
            'top10_accuracy': metrics['top10_acc'] / metrics['total_tokens'],
            'evaluated_samples': metrics['total_samples'],
            'evaluated_tokens': metrics['total_tokens']
        }

if __name__ == "__main__":
    evaluator = ModelEvaluator(
        checkpoint_path="/home/u2024000980/sanliwan/chartransformer_1/char5.6/char_transformer_epoch9.pt",
        vocab_path=Config.vocab_path,
        test_data_path="/home/u2024000980/sanliwan/chartransformer_1/data/fanticeshiji.txt"
    )
    
    results = evaluator.evaluate()
    
    print("\n评估结果：")
    print(f"- 困惑度 (PPL): {results['perplexity']:.2f}")
    print(f"- 平均排名百分比: {results['avg_rank_percent']*100:.2f}%")
    print(f"- 平均排名: {results['avg_rank']:.1f}")
    print(f"- Top1准确率: {results['top1_accuracy']*100:.2f}%")
    print(f"- Top5准确率: {results['top5_accuracy']*100:.2f}%")
    print(f"- Top10准确率: {results['top10_accuracy']*100:.2f}%")
    print(f"- 评估样本数: {results['evaluated_samples']}")
    print(f"- 评估Token数: {results['evaluated_tokens']}")