import os
from IPython.display import Audio, display
import soundfile as sf

SR = 44100
idx = 6  # Change this to play a different sample

# Paths to each stem folder, can be used for both train and test sets
bass_path   = "_____/bass"
drums_path  = "_____/drums"
vocals_path = "_____/vocals"
others_path = "_____/others"

# Get sorted lists of files
bass_files   = sorted([f for f in os.listdir(bass_path) if f.endswith(".wav")])
drums_files  = sorted([f for f in os.listdir(drums_path) if f.endswith(".wav")])
vocals_files = sorted([f for f in os.listdir(vocals_path) if f.endswith(".wav")])
others_files = sorted([f for f in os.listdir(others_path) if f.endswith(".wav")])

# Load and play bass
bass_file = os.path.join(bass_path, bass_files[idx])
bass_audio, sr = sf.read(bass_file)
if bass_audio.ndim > 1:
    bass_audio = bass_audio.mean(axis=1)
print("Bass:", bass_files[idx])
display(Audio(bass_audio, rate=sr))

# Load and play drums
drums_file = os.path.join(drums_path, drums_files[idx])
drums_audio, sr = sf.read(drums_file)
if drums_audio.ndim > 1:
    drums_audio = drums_audio.mean(axis=1)
print("Drums:", drums_files[idx])
display(Audio(drums_audio, rate=sr))

# Load and play vocals
vocals_file = os.path.join(vocals_path, vocals_files[idx])
vocals_audio, sr = sf.read(vocals_file)
if vocals_audio.ndim > 1:
    vocals_audio = vocals_audio.mean(axis=1)
print("Vocals:", vocals_files[idx])
display(Audio(vocals_audio, rate=sr))

# Load and play others
others_file = os.path.join(others_path, others_files[idx])
others_audio, sr = sf.read(others_file)
if others_audio.ndim > 1:
    others_audio = others_audio.mean(axis=1)
print("Others:", others_files[idx])
display(Audio(others_audio, rate=sr))
