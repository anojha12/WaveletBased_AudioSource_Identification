import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from raw_config import DEVICE, BATCH_SIZE, EPOCHS, LEARNING_RATE, TARGET_SR
from data.raw_dataset import RawWaveformDataset, get_file_label_list
from models.raw_wave_cnn import RawWaveCNN
from utils.train_utils import train_one_epoch, validate  # import from utils

train_losses = []
val_losses = []

def main_training(root_dir):
    # Load file paths and labels
    file_paths, labels = get_file_label_list(root_dir)
    n = len(file_paths)
    idxs = np.random.permutation(n)

    split = int(n * 0.9)
    train_idx, val_idx = idxs[:split], idxs[split:]

    train_files = [file_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_files = [file_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # Create datasets and dataloaders
    train_ds = RawWaveformDataset(train_files, train_labels, target_sr=TARGET_SR)
    val_ds = RawWaveformDataset(val_files, val_labels, target_sr=TARGET_SR)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, optimizer, and loss
    model = RawWaveCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # Train and validate using utils functions
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, _, _ = validate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_rawwave_cnn.pth")
            print("Saved best model.")

    # Save loss history
    np.save("train_losses_raw.npy", np.array(train_losses))
    np.save("val_losses_raw.npy", np.array(val_losses))


if __name__ == "__main__":
    main_training("/MUSDB18-train")
