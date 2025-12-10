import numpy as np

def add_awgn(x, snr_db):
    """
    Add AWGN noise to a 1D audio signal.

    Args:
        x: numpy array of audio samples (float32)
        snr_db: target SNR in dB

    Returns:
        noisy audio
    """
    sig_power = np.mean(x**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), x.shape)
    return x + noise
