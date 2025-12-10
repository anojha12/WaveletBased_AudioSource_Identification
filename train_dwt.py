import numpy as np
from torch.utils.data import DataLoader
import torch
from data.dwt_dataset import get_file_label_list, WaveletDataset
from models.wavelet_cnn import WaveletCNN
from utils.train_utils import train_one_epoch, validate
from dwt_config import *

def main_training(root_dir):
    file_paths, labels = get_file_label_list(root_dir)

    n = len(file_paths)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    split = int(n * 0.9)

    train_files = [file_paths[i] for i in idxs[:split]]
    train_labels = [labels[i] for i in idxs[:split]]
    val_files = [file_paths[i] for i in idxs[split:]]
    val_labels = [labels[i] for i in idxs[split:]]

    train_ds = WaveletDataset(train_files, train_labels)
    val_ds = WaveletDataset(val_files, val_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = WaveletCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, _, _ = validate(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_wavelet_cnn.pth")
            print("Saved best model.")

if __name__ == "__main__":
    main_training("/MUSDB18-train")
