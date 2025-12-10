import torch
import numpy as np
import pywt
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict_wavercnn(model, audio, max_len=300032):
    """Predict single label using WaveRCNN"""
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]

    x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    return torch.argmax(logits, dim=1).item()


def compute_dwt(audio, wavelet, level):
    """Compute multi-level DWT and flatten coefficients"""
    coeffs = pywt.wavedec(audio, wavelet=wavelet, level=level)
    return np.concatenate([c for c in coeffs], axis=-1)


def predict_dwtcnn(model, audio, wavelet, level, target_size=4096):
    """Predict single label using DWT-CNN"""
    dwt_vec = compute_dwt(audio, wavelet, level)
    x = torch.tensor(dwt_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = F.interpolate(x, size=target_size).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
    return torch.argmax(logits, dim=1).item()
