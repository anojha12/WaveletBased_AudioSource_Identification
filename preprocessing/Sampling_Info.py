import librosa

# Path to one audio file from your dataset
audio_path = "______/MUSDB18-train/bass/A Classic Education - NightOwl_bass.wav"

# Load with original sampling rate (do NOT resample)
wav, sr = librosa.load(audio_path, sr=None)

print("Sampling Rate =", sr)
print("Number of samples =", len(wav))
print("Duration (sec) =", len(wav) / sr)