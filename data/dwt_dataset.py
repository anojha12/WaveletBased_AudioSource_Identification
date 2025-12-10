import os
import glob
import numpy as np
import librosa
import pywt
import torch
from torch.utils.data import Dataset
from dwt_config import STEMS, MAX_LEN, DWT_LEVELS

def get_file_label_list(root_dir):
    file_paths = []
    labels = []

    for sub in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        stems_in_folder = sub.split("_")
        lab = np.zeros(len(STEMS), dtype=np.float32)
        for i, s in enumerate(STEMS):
            if s in stems_in_folder:
                lab[i] = 1.0

        wav_files = glob.glob(os.path.join(sub_path, "*.wav"))
        for w in wav_files:
            file_paths.append(w)
            labels.append(lab.copy())

    return file_paths, labels


class WaveletDataset(Dataset):
    def __init__(self, file_paths, labels, max_len=MAX_LEN, dwt_levels=DWT_LEVELS):
        self.file_paths = file_paths
        self.labels = labels
        self.max_len = max_len
        self.dwt_levels = dwt_levels

    def __len__(self):
        return len(self.file_paths)

    def dwt_downsample(self, x):
        for _ in range(self.dwt_levels):
            x, _ = pywt.dwt(x, "haar")
        return x

    def pad_or_trim(self, x):
        if len(x) > self.max_len:
            return x[:self.max_len]
        return np.pad(x, (0, self.max_len - len(x)))

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        wav, sr = librosa.load(path, sr=None)
        wav = self.dwt_downsample(wav)
        wav = self.pad_or_trim(wav)

        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.float32)

        return x, y
