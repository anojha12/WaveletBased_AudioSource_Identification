import torch

STEMS = ["vocals", "drums", "bass", "other"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 150
LEARNING_RATE = 1e-3
DWT_LEVELS = 5
MAX_LEN = 9376
