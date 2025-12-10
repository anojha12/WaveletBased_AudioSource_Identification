import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from raw_config import DEVICE, BATCH_SIZE
from data.raw_dataset import RawWaveformDataset, get_file_label_list
from models.raw_wave_cnn import RawWaveCNN


def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='samples', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)

    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")
    return all_preds, all_labels


def main_test(test_root_dir, model_path="best_rawwave_cnn.pth"):
    file_paths, labels = get_file_label_list(test_root_dir)
    test_ds = RawWaveformDataset(file_paths, labels)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = RawWaveCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    preds, labels = evaluate_model(model, test_loader, DEVICE)

    print("\nFirst 5 predictions vs labels:")
    for p, l in zip(preds[:5], labels[:5]):
        print(p, l)


if __name__ == "__main__":
    main_test("/MUSDB18-test")
