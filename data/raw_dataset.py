import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from raw_config import STEMS, TARGET_SR

def get_file_label_list(root_dir):
    file_paths, labels = [], []
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


class RawWaveformDataset(Dataset):
    def __init__(self, file_paths, labels, target_sr=TARGET_SR):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sr = target_sr

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        wav, sr = librosa.load(path, sr=None)
        if sr != self.target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)

        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y
