import os
import torch
import numpy as np
from model import TransformerLM
from data_loader import CharVocab
from utils import compute_n95, get_top_n_candidates
from config import *

# ================== 配置 ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "/data/zhouyufan/llmpro/models/models.pth/model_epoch_2.pth"
TEST_TEXT = "我想去□园玩耍，可是我的好朋友不想陪我去。"
SAVE_LOG_PATH = "/data/zhouyufan/llmpro/test_results.txt"
TOP_K = 100
NUM_TEST = 5
# ========================================

def safe_int(val, name):
    if not isinstance(val, int):
        val = int(val)
        print(f"⚙️ 参数 {name} 自动转换为 int: {val}")
    return val

def test_model():
    print("🚀 启动测试流程...")
    print("=" * 60)
    print(f"✅ 使用设备: {DEVICE}")

    # ---------------- 词表 ----------------
    try:
        vocab = CharVocab(DICTIONARY_PATH)
        print(f"✅ 成功加载词表文件: {DICTIONARY_PATH}")
        print(f"✅ 词表大小: {vocab.vocab_size}")
    except Exception as e:
        print(f"❌ 词表加载失败: {e}")
        return

    if not hasattr(vocab, "id_to_token"):
        print("⚙️ 检测到 CharVocab 缺少 id_to_token，自动构建中...")
        vocab.id_to_token = {idx: char for char, idx in vocab.char_to_idx.items()}
        print("✅ 已自动生成 id_to_token")
        sample_tokens = list(vocab.id_to_token.values())[:20]
        print(f"✅ 示例 token 前 20 个: {sample_tokens}")

    # ---------------- 模型 ----------------
    try:
        model = TransformerLM(
            vocab_size=safe_int(vocab.vocab_size, "vocab_size"),
            embed_dim=safe_int(EMBED_DIM, "EMBED_DIM"),
            hidden_dim=safe_int(HIDDEN_DIM, "HIDDEN_DIM"),
            num_layers=safe_int(NUM_LAYERS, "NUM_LAYERS"),
            num_heads=safe_int(NUM_HEADS, "NUM_HEADS"),
            num_queries=safe_int(SPARSE_QUERIES, "SPARSE_QUERIES"),
            dropout=float(DROPOUT)
        ).to(DEVICE)
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        return

    # ---------------- 加载权重 ----------------
    try:
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ 已加载模型参数文件: {CKPT_PATH}")
    except Exception as e:
        print(f"❌ 模型参数加载失败: {e}")
        return

    model.eval()
    print("✅ 模型模式: eval\n")

    # ---------------- 测试 ----------------
    print(f"📝 测试文本: {TEST_TEXT}")

    tokens = [vocab.char_to_idx[SOS_TOKEN]] + vocab.encode(TEST_TEXT) + [vocab.char_to_idx[EOS_TOKEN]]
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
    else:
        tokens += [vocab.char_to_idx[PAD_TOKEN]] * (MAX_SEQ_LEN - len(tokens))

    n95_list = []
    log_lines = []

    for run in range(NUM_TEST):
        masked_tokens = tokens.copy()
        mask_positions = [0] * len(tokens)
        valid_positions = [i for i, t in enumerate(tokens) if t not in [
            vocab.char_to_idx[SOS_TOKEN],
            vocab.char_to_idx[EOS_TOKEN],
            vocab.char_to_idx[PAD_TOKEN]
        ]]
        num_masks = max(1, int(len(valid_positions) * 0.2))
        mask_indices = np.random.choice(valid_positions, num_masks, replace=False).tolist()

        for pos in mask_indices:
            masked_tokens[pos] = vocab.char_to_idx[MASK_TOKEN]
            mask_positions[pos] = 1

        input_tensor = torch.tensor([masked_tokens], device=DEVICE)
        labels = torch.tensor(tokens, device=DEVICE)
        mask_tensor = torch.tensor(mask_positions, device=DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            n95, _ = compute_n95(logits[0], labels, mask_tensor)
            n95_list.append(n95.float().mean().item())
            # 直接调用旧版 get_top_n_candidates
            candidates_all = get_top_n_candidates(logits[0], mask_tensor, vocab)
            # 取前 TOP_K
            candidates = []
            for chars, probs in candidates_all:
                chars_top = chars[:TOP_K]
                probs_top = probs[:TOP_K]
                candidates.append((chars_top, probs_top))

        print(f"\n===== 测试轮次 {run + 1} =====")
        log_lines.append(f"\n===== 测试轮次 {run + 1} =====")
        for i, (chars, probs) in enumerate(candidates):
            print(f"Mask {i + 1}: Top-{TOP_K} 候选 - {chars}, 概率 - {probs}")
            log_lines.append(f"Mask {i + 1}: Top-{TOP_K} 候选 - {chars}, 概率 - {probs}")

    # ---------------- N95 ----------------
    n95_mean = np.mean(n95_list)
    n95_std = np.std(n95_list)
    print(f"\n✅ 测试完成！")
    print(f"N95 均值: {n95_mean:.2f}, 标准差: {n95_std:.2f}")
    log_lines.append(f"\n✅ 测试完成！\nN95 均值: {n95_mean:.2f}, 标准差: {n95_std:.2f}")

    # ---------------- 保存日志 ----------------
    with open(SAVE_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"✅ 日志已保存到: {SAVE_LOG_PATH}")


if __name__ == "__main__":
    test_model()
