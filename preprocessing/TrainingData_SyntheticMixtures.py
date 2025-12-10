import os
import itertools
import soundfile as sf
import numpy as np

# Base path where original stems are stored
base_path = "_____/MUSDB18-train/"

# Original stem folders
stems = ["bass", "vocals", "drums", "others"]

# Output path for synthetic mixtures
output_base = os.path.join(base_path, "synthetic_mixtures")
os.makedirs(output_base, exist_ok=True)

# Get sorted list of files for each stem
stem_files = {}
for stem in stems:
    folder = os.path.join(base_path, stem)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".wav")])
    stem_files[stem] = files

# Function to load audio as mono
def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr

# Function to save audio
def save_audio(audio, sr, out_path):
    sf.write(out_path, audio, sr)

# Generate all 2-stem and 3-stem combinations
for r in [2, 3]:
    combos = list(itertools.combinations(stems, r))
    for combo in combos:
        combo_name = "_".join(combo)
        out_folder = os.path.join(output_base, combo_name)
        os.makedirs(out_folder, exist_ok=True)
        
        # Number of files (assuming all stems have same filenames)
        num_files = len(stem_files[combo[0]])
        
        for idx in range(num_files):
            audios = []
            sr = None
            
            # Load each stem in the combination
            for stem in combo:
                file_path = os.path.join(base_path, stem, stem_files[stem][idx])
                audio, sr = load_audio(file_path)
                audios.append(audio)
            
            # Check lengths and truncate to minimum length
            lengths = [a.shape[0] for a in audios]
            min_len = min(lengths)
            if len(set(lengths)) > 1:
                print(f"Truncating track {stem_files[combo[0]][idx]} for combination {combo_name} "
                      f"from lengths {lengths} to minimum length {min_len}")
            audios = [a[:min_len] for a in audios]
            
            # Sum to create mixture
            mixture = np.sum(audios, axis=0)
            
            # Normalize to avoid clipping
            mixture = mixture / np.max(np.abs(mixture))
            
            # Save synthetic mixture
            out_file = os.path.join(out_folder, stem_files[combo[0]][idx])  # use first stem's filename
            save_audio(mixture, sr, out_file)

print("All synthetic 2-stem and 3-stem mixtures generated successfully!")