import os
import musdb
import soundfile as sf

# Paths
musdb_root = "______/MUSDB18/MUSDB18-7"
output_root = "______/MUSDB18/MUSDB18-train"

# Create folders for stems
stems = ["vocals", "drums", "bass", "others"]
for stem in stems:
    os.makedirs(os.path.join(output_root, stem), exist_ok=True)

# Load training dataset
mus_train = musdb.DB(root=musdb_root, subsets="train")

# Iterate over all tracks
for track in mus_train.tracks:
    print(f"Processing track: {track.name}")
    # Save each stem individually
    for stem_name in ["vocals", "drums", "bass", "accompaniment"]:
        audio = track.targets[stem_name].audio  # shape: (samples, channels)
        
        # Map 'accompaniment' to 'others'
        out_stem_name = "others" if stem_name == "accompaniment" else stem_name
        
        # File path to save
        out_file = os.path.join(output_root, out_stem_name, f"{track.name}_{stem_name}.wav")
        
        # Save as wav
        sf.write(out_file, audio, track.rate)

print("All tracks separated!")
