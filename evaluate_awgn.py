import numpy as np
import torch
import librosa
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.noise_utils import add_awgn
from utils.predict_utils import predict_wavercnn, predict_dwtcnn
from data.raw_dataset import get_file_label_list

from raw_config import MAX_LEN, DEVICE as RAW_DEVICE
from dwt_config import DWT_WAVELET, DWT_LEVEL, DWT_TARGET_LEN, DEVICE as DWT_DEVICE

from models.raw_wave_cnn import RawWaveCNN
from models.wavelet_cnn import WaveletCNN

def test_under_noise(wavercnn_model, dwt_model, test_files, test_labels):
    SNR_LEVELS = [30, 40, 50]
    results = {}

    for snr in SNR_LEVELS:
        y_true, y_pred_wave, y_pred_dwt = [], [], []

        for audio_path, label in zip(test_files, test_labels):
            audio, sr = librosa.load(audio_path, sr=44100)
            noisy_audio = add_awgn(audio, snr)

            # Raw model prediction
            y_pred_wave.append(predict_wavercnn(wavercnn_model, noisy_audio, max_len=MAX_LEN))

            # DWT model prediction
            y_pred_dwt.append(predict_dwtcnn(
                dwt_model, noisy_audio, wavelet=DWT_WAVELET, level=DWT_LEVEL, target_len=DWT_TARGET_LEN
            ))

            # Convert one-hot label to integer
            if isinstance(label, (np.ndarray, list)):
                y_true.append(int(np.argmax(label)))
            else:
                y_true.append(label)

        # compute metrics
        acc_w = accuracy_score(y_true, y_pred_wave)
        acc_d = accuracy_score(y_true, y_pred_dwt)
        p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred_wave, average='macro')
        p_d, r_d, f_d, _ = precision_recall_fscore_support(y_true, y_pred_dwt, average='macro')

        results[snr] = {
            "WaveRCNN": {"Accuracy": acc_w, "Precision": p_w, "Recall": r_w, "F1": f_w},
            "DWT-CNN": {"Accuracy": acc_d, "Precision": p_d, "Recall": r_d, "F1": f_d}
        }

    return results


if __name__ == "__main__":
    # Load models
    wavercnn_model = RawWaveCNN().to(RAW_DEVICE)
    wavercnn_model.load_state_dict(torch.load("best_rawwave_cnn.pth", map_location=RAW_DEVICE))
    wavercnn_model.eval()

    dwt_model = WaveletCNN().to(DWT_DEVICE)
    dwt_model.load_state_dict(torch.load("best_wavelet_cnn.pth", map_location=DWT_DEVICE))
    dwt_model.eval()

    # Load test files
    test_files, test_labels = get_file_label_list("/MUSDB18-test")

    # Evaluate
    results = test_under_noise(wavercnn_model, dwt_model, test_files, test_labels)

    # Print
    for snr, metrics in results.items():
        print(f"\nSNR = {snr} dB")
        print("WaveRCNN:", metrics["WaveRCNN"])
        print("DWT-CNN:", metrics["DWT-CNN"])
