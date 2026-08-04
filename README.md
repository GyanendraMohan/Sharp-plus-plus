# SHARP++

**Residual CNN + CBAM Attention + Bi-LSTM for WiFi Human Activity Recognition**

A drop-in classifier upgrade for the [SHARP](https://github.com/francescamen/SHARP) framework.
The entire SHARP preprocessing pipeline (phase sanitization, sparse channel estimation,
micro-Doppler generation) is kept **unchanged**. Only the classification stage is replaced.

---

## Architecture

```
Input Spectrogram  (time × freq × 4 antennas)
        │
  Stem Conv2D(64) + BN + ReLU
        │
  ResBlock(64)  + MaxPool(2×2)        ←  local motion features
        │
  ResBlock(128) + MaxPool(2×2)
        │
  ResBlock(256)
        │
  CBAM  (Channel + Spatial Attention)  ←  suppress noise / highlight Doppler
        │
  Reshape  →  (time/4,  freq/4 × 256)
        │
  Bi-LSTM(128)  ×2                    ←  temporal gait periodicity
        │
  GlobalAveragePooling1D
        │
  Dense(256, ReLU)  →  Dropout(0.5)
        │
  Dense(num_classes, Softmax)
        │
  Activity Prediction
```

### Key improvements over original SHARP classifier

| Feature | Original SHARP | SHARP++ |
|---------|---------------|---------|
| Feature extractor | Inception-style CNN | Residual CNN (ResNet-style) |
| Attention | None | CBAM (channel + spatial) |
| Temporal modeling | None | Bidirectional LSTM × 2 |
| Antenna fusion | Per-antenna, majority vote | Multi-channel input (joint) |
| Gradient flow | Standard | Skip connections (no vanishing) |

---

## Prerequisites

Same environment as the original SHARP:

```bash
source ../.venv/bin/activate        # or your existing virtual environment
pip install scikit-learn            # if not already installed
```

> **Note:** `tensorflow`, `numpy`, `pickle` are already installed in the SHARP environment.

---

## Quick Start

### 1. Verify preprocessing is complete

The `doppler_traces/` directory must already contain the windowed dataset created by
SHARP's `CSI_doppler_create_dataset_train.py`. For each session you should see:

```
Python_code/doppler_traces/S1a/
    train_antennas_E,L,W,R,J/     ← window .txt files
    val_antennas_E,L,W,R,J/
    test_antennas_E,L,W,R,J/
    labels_train_antennas_E,L,W,R,J.txt
    files_train_antennas_E,L,W,R,J.txt
    ...
```

### 2. Train

```bash
cd SHARP_plus_plus

# Train on sessions S1a, S1b, S1c — activities E, L, W, R, J
python train.py \
    ../Python_code/doppler_traces/ \
    S1a,S1b,S1c \
    E,L,W,R,J \
    ./outputs/sharp_pp

# With custom hyperparameters
python train.py \
    ../Python_code/doppler_traces/ \
    S1a,S1b,S1c,S2a,S2b \
    E,L,W,R,J \
    ./outputs/sharp_pp \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --patience 10
```

Output files written to `./outputs/`:

| File | Contents |
|------|----------|
| `sharp_pp_E,L,W,R,J_network.keras` | Best model (by val accuracy) |
| `sharp_pp_E,L,W,R,J_history.json` | Loss & accuracy curves |
| `sharp_pp_E,L,W,R,J_test_results.txt` | Pickled metrics dict |

### 3. Evaluate

```bash
# Test on same sessions
python test.py \
    ../Python_code/doppler_traces/ \
    S1a,S1b,S1c \
    E,L,W,R,J \
    ./outputs/sharp_pp_E,L,W,R,J_network.keras

# Cross-session evaluation (train S1*, test S2*)
python test.py \
    ../Python_code/doppler_traces/ \
    S2a,S2b \
    E,L,W,R,J \
    ./outputs/sharp_pp_E,L,W,R,J_network.keras \
    --split complete
```

---

## Input Format

The model expects the windowed Doppler spectrograms already produced by SHARP.
Each `.txt` window file is a **pickled numpy array** with shape:

```
(num_antennas, freq_bins, time_steps)
```

`dataset_loader.py` transposes this to `(time_steps, freq_bins, num_antennas)` — the
4-channel input tensor consumed by the model.

**Actual dimensions** are auto-detected from the first sample; no hardcoded sizes needed.

---

## Troubleshooting

### `FileNotFoundError: Label file not found`

The windowed dataset has not been created for that session/activity combination.
Run SHARP's `CSI_doppler_create_dataset_train.py` first:

```bash
cd Python_code
python CSI_doppler_create_dataset_train.py \
    doppler_traces/ S1a 51 1 100 5 E,L,W,R,J 4
```

### File paths don't exist (project moved to a different machine)

Add `--fix_paths` to re-map the stored paths to the current `doppler_dir`:

```bash
python train.py ../Python_code/doppler_traces/ S1a E,L,W,R,J ./outputs/sharp_pp --fix_paths
```

### Out of memory on GPU

Reduce batch size:
```bash
python train.py ... --batch_size 16
```

Or reduce LSTM/Dense size:
```bash
python train.py ... --lstm_units 64 --dense_units 128
```

---

## Module Reference

| File | Description |
|------|-------------|
| `model.py` | `build_sharp_plus_plus()`, `residual_block()`, `cbam_block()` |
| `dataset_loader.py` | `load_multi_antenna()`, `build_tf_dataset()`, `load_sharp_file_lists()` |
| `train.py` | End-to-end training + evaluation CLI |
| `test.py` | Standalone inference + metrics CLI |
| `outputs/` | Saved models, histories, and result pickles |

---

## Citation

If you use SHARP++, please cite the original SHARP paper:

```
@article{meneghello2022sharp,
  title={SHARP: Environment and Person Independent Activity Recognition with Commodity IEEE 802.11 Access Points},
  author={Meneghello, Francesca and others},
  journal={IEEE Transactions on Mobile Computing},
  year={2022}
}
```
