import os
from IPython.display import Audio, display
import soundfile as sf

SR = 44100
idx = 5  # Change this to play a different sample

# Base folder containing all synthetic mixture combinations
synthetic_base = "" # Replace with specific path for train or test to play samples from each

# Paths to each combination folder (explicitly)
bass_vocals_path       = os.path.join(synthetic_base, "bass_vocals")
bass_drums_path        = os.path.join(synthetic_base, "bass_drums")
bass_others_path       = os.path.join(synthetic_base, "bass_others")
vocals_drums_path      = os.path.join(synthetic_base, "vocals_drums")
vocals_others_path     = os.path.join(synthetic_base, "vocals_others")
drums_others_path      = os.path.join(synthetic_base, "drums_others")
bass_vocals_drums_path = os.path.join(synthetic_base, "bass_vocals_drums")
bass_vocals_others_path= os.path.join(synthetic_base, "bass_vocals_others")
bass_drums_others_path = os.path.join(synthetic_base, "bass_drums_others")
vocals_drums_others_path= os.path.join(synthetic_base, "vocals_drums_others")

# Function to load audio as mono
def load_mono_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr

# Get sorted file lists
bass_vocals_files        = sorted([f for f in os.listdir(bass_vocals_path) if f.endswith(".wav")])
bass_drums_files         = sorted([f for f in os.listdir(bass_drums_path) if f.endswith(".wav")])
bass_others_files        = sorted([f for f in os.listdir(bass_others_path) if f.endswith(".wav")])
vocals_drums_files       = sorted([f for f in os.listdir(vocals_drums_path) if f.endswith(".wav")])
vocals_others_files      = sorted([f for f in os.listdir(vocals_others_path) if f.endswith(".wav")])
drums_others_files       = sorted([f for f in os.listdir(drums_others_path) if f.endswith(".wav")])
bass_vocals_drums_files  = sorted([f for f in os.listdir(bass_vocals_drums_path) if f.endswith(".wav")])
bass_vocals_others_files = sorted([f for f in os.listdir(bass_vocals_others_path) if f.endswith(".wav")])
bass_drums_others_files  = sorted([f for f in os.listdir(bass_drums_others_path) if f.endswith(".wav")])
vocals_drums_others_files= sorted([f for f in os.listdir(vocals_drums_others_path) if f.endswith(".wav")])

# Play bass_vocals
file_path = os.path.join(bass_vocals_path, bass_vocals_files[idx])
audio, sr = load_mono_audio(file_path)
print("Bass + Vocals:", bass_vocals_files[idx])
display(Audio(audio, rate=sr))

# Play bass_drums
file_path = os.path.join(bass_drums_path, bass_drums_files[idx])
audio, sr = load_mono_audio(file_path)
print("Bass + Drums:", bass_drums_files[idx])
display(Audio(audio, rate=sr))

# Play bass_others
file_path = os.path.join(bass_others_path, bass_others_files[idx])
audio, sr = load_mono_audio(file_path)
print("Bass + Others:", bass_others_files[idx])
display(Audio(audio, rate=sr))

# Play vocals_drums
file_path = os.path.join(vocals_drums_path, vocals_drums_files[idx])
audio, sr = load_mono_audio(file_path)
print("Vocals + Drums:", vocals_drums_files[idx])
display(Audio(audio, rate=sr))

# Play vocals_others
file_path = os.path.join(vocals_others_path, vocals_others_files[idx])
audio, sr = load_mono_audio(file_path)
print("Vocals + Others:", vocals_others_files[idx])
display(Audio(audio, rate=sr))

# Play drums_others
file_path = os.path.join(drums_others_path, drums_others_files[idx])
audio, sr = load_mono_audio(file_path)
print("Drums + Others:", drums_others_files[idx])
display(Audio(audio, rate=sr))

# Play bass_vocals_drums
file_path = os.path.join(bass_vocals_drums_path, bass_vocals_drums_files[idx])
audio, sr = load_mono_audio(file_path)
print("Bass + Vocals + Drums:", bass_vocals_drums_files[idx])
display(Audio(audio, rate=sr))

# Play bass_vocals_others
file_path = os.path.join(bass_vocals_others_path, bass_vocals_others_files[idx])
audio, sr = load_mono_audio(file_path)
print("Bass + Vocals + Others:", bass_vocals_others_files[idx])
display(Audio(audio, rate=sr))

# Play bass_drums_others
file_path = os.path.join(bass_drums_others_path, bass_drums_others_files[idx])
audio, sr = load_mono_audio(file_path)
print("Bass + Drums + Others:", bass_drums_others_files[idx])
display(Audio(audio, rate=sr))

# Play vocals_drums_others
file_path = os.path.join(vocals_drums_others_path, vocals_drums_others_files[idx])
audio, sr = load_mono_audio(file_path)
print("Vocals + Drums + Others:", vocals_drums_others_files[idx])
display(Audio(audio, rate=sr))
