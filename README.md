# Musical Source Identification with WaveRCNN and DWT-CNN

This project implements musical source identification using **raw waveform CNN (WaveRCNN)** and **wavelet-transformed CNN (DWT-CNN)** models. 
It includes preprocessing scripts to create synthetic mixtures, training scripts, evaluation scripts, and testing under noisy conditions.

This project uses the MUSDB18 dataset, available here: https://zenodo.org/records/1117372
After downloading, place the dataset in the project root
**Setup:**
```bash
# After downloading, place the dataset in the project root
├── MUSDB18/
│   ├── train/
│   └── test/
```

## Pipeline

Important: Preprocessing is Required First

The preprocessing step must be completed before training. It generates the synthetic mixtures that both models rely on.

## Usage
### Step 1: Preprocessing

#### 1.1 Inspect Raw Audio (Optional)
Explore the original dataset samples:
```bash
python preprocessing/Data_Samples.py
python preprocessing/Sampling_Info.py
```

#### 1.2 Segment Audio Stems
Process and segment raw stems into fixed-length chunks:
```bash
# Training data
python preprocessing/TrainData_preprocessing.py

# Test data
python preprocessing/TestData_preprocessing.py
```

#### 1.3 Generate Synthetic Mixtures
Create synthetic mixtures by combining different stems:
```bash
# Training mixtures
python preprocessing/TrainingData_SyntheticMixtures.py

# Test mixtures
python preprocessing/TestData_SyntheticMixtures.py
```

#### 1.4 Verify Mixtures (Optional)
Listen to generated synthetic mixtures:
```bash
python preprocessing/SyntheticMixture_Samples.py
```

---

### Step 2: Training

Train both model architectures on the preprocessed data:

#### WaveRCNN (Raw Waveform Model)
```bash
python train_raw.py --data_dir preprocessing/synthetic_train_data
```

#### DWT-CNN (Wavelet Transform Model)
```bash
python train_dwt.py --data_dir preprocessing/synthetic_train_data
```

---

### Step 3: Evaluation

#### 3.1 Clean Test Data Evaluation
Assess model performance on clean synthetic mixtures:
```bash
# Evaluate WaveRCNN
python evaluate_raw.py --data_dir preprocessing/synthetic_test_data

# Evaluate DWT-CNN
python evaluate_dwt.py --data_dir preprocessing/synthetic_test_data
```

#### 3.2 Robustness Testing (AWGN Noise)
Test model robustness under Additive White Gaussian Noise:
```bash
python test_awgn.py --data_dir preprocessing/synthetic_test_data
```
