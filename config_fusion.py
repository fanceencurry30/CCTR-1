# Configuration parameter file

# Fusion model parameters
FEATURE_DIM = 100  # Dimension of top 100 candidate characters
NUM_HEADS = 8      # Number of attention heads
HIDDEN_DIM = 256   # Hidden layer dimension of feed-forward network
DROPOUT = 0.1      # Dropout ratio

# Training parameters
BATCH_SIZE = 128 #256
EPOCHS = 1000

LEARNING_RATE = 5e-6
LEARNING_RATE2 = 5e-6

# File paths
FUSION_MODEL_SAVE_PATH = "models_60p/fusion_model"