import torch

STEMS = ["vocals", "drums", "bass", "other"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
EPOCHS = 150
LEARNING_RATE = 1e-3
TARGET_SR = 22050
