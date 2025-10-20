# 模型参数
VOCAB_SIZE = 9633  # 词汇表大小，根据实际情况从char_to_idx.json加载
EMBED_DIM = 512     # 嵌入维度
HIDDEN_DIM = 3072   # 前馈网络隐藏维度
NUM_LAYERS = 24     # Transformer层数
NUM_HEADS = 8       # 注意力头数
SPARSE_QUERIES = 64 # 稀疏注意力查询向量数量
DROPOUT = 0.1       # Dropout比例

# 训练参数
BATCH_SIZE = 128     # 批次大小
EPOCHS = 50         # 训练轮数
LEARNING_RATE = 1e-4 # 学习率
MAX_SEQ_LEN = 128   # 最大序列长度
NOISE_P = 0.02      # 非掩码字符加噪比例

# 数据路径
DICTIONARY_PATH = "/data/zhouyufan/llmpro/data/char_to_idx.json"  # 字典文件路径
TRAIN_DATA_PATH = "/data/zhouyufan/llmpro/data/train.txt"         # 训练数据路径
MODEL_SAVE_PATH = "/data/zhouyufan/llmpro/models/models.pth"  # 模型保存路径
CACHE_PATH = "/data/zhouyufan/llmpro/data/train_cache.pkl"        # 预处理数据缓存路径

# 特殊token
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "□"    # 空白字符标记
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"