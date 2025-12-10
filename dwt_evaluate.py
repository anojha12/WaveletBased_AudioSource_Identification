import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data.dwt_dataset import WaveletDataset, get_file_label_list
from models.wavelet_cnn import WaveletCNN
from dwt_config import DEVICE, BATCH_SIZE


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='samples', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)

    return acc, prec, rec, f1, all_preds, all_labels


def main_test(test_root_dir, model_path="best_wavelet_cnn.pth"):
    # Load test dataset
    file_paths, labels = get_file_label_list(test_root_dir)
    test_ds = WaveletDataset(file_paths, labels)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Evaluating {len(test_ds)} test audio files...")

    # Load model
    model = WaveletCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Loaded model from {model_path}")

    # Run evaluation
    acc, prec, rec, f1, preds, labels = evaluate_model(model, test_loader, DEVICE)

    # Output results
    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nFirst 5 predictions vs labels:")
    for p, l in zip(preds[:5], labels[:5]):
        print("Pred:", p, "Label:", l)

    return preds, labels


if __name__ == "__main__":
    # Example usage:
    # python evaluate.py
    TEST_FOLDER = "musdb18hq/test"
    MODEL_FILE = "best_wavelet_cnn.pth"
    main_test(TEST_FOLDER, MODEL_FILE)
