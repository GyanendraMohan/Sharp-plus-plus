# SHARP++ 5-Class LOEO — Fold 1 Results

## Experiment Overview

| Property                     | Value                                                                  |
| :--------------------------- | :--------------------------------------------------------------------- |
| **Model**              | SHARP++ (ResCNN–CBAM–BiLSTM)                                         |
| **Task**               | 5-Class Human Activity Recognition from Wi-Fi CSI Doppler Spectrograms |
| **Protocol**           | Leave-One-Environment-Out (LOEO)                                       |
| **Fold**               | 1 of 3                                                                 |
| **Test Environment**   | 🛏️**Bedroom** (S1a, S1b, S1c, S2a, S2b, S3a, S4a, S4b, S5a)    |
| **Train Environments** | 🛋️ Living Room (S6a, S6b) + 🔬 Laboratory (S7a)                      |

---

## Dataset Split

|        Split        | Windows | Description                                                                       |
| :------------------: | ------: | :-------------------------------------------------------------------------------- |
|   **Train**   |  38,876 | Independent single-antenna windows from Living Room + Laboratory (75% time split) |
| **Validation** |  12,180 | Independent single-antenna windows from Living Room + Laboratory (25% time split) |
|    **Test**    |  51,757 | 4-antenna grouped windows from Bedroom (100% — completely held out)              |

> **Note**: This is the hardest training scenario in the LOEO protocol — the model is trained on only **3 sessions** (S6a, S6b, S7a) from 2 environments, yet must generalize to **9 sessions** of a completely unseen residential bedroom.

---

## Classes

| Index | Class   | SHARP Code | Description                   |
| :---: | :------ | :--------: | :---------------------------- |
|   0   | Empty   |   `E`   | No human presence in the room |
|   1   | Sitting |   `L`   | Seated somewhere in the room  |
|   2   | Walking |   `W`   | Walking around the room       |
|   3   | Running |   `R`   | Running around the room       |
|   4   | Jumping |   `J`   | Jumping in place              |

---

## Results Summary

| Metric                     |      Value      |
| :------------------------- | :--------------: |
| **Overall Accuracy** | **82.50%** |
| **Macro Precision**  |      81.94%      |
| **Macro Recall**     |      81.53%      |
| **Macro F1-Score**   | **79.45%** |

---

## Per-Class Performance

| Class   | Precision |  Recall |         F1-Score | Support |
| :------ | --------: | ------: | ---------------: | ------: |
| Empty   |    99.29% |  99.94% | **99.61%** |  11,725 |
| Sitting |    94.82% | 100.00% | **97.34%** |  11,913 |
| Walking |    65.11% |  81.38% | **72.34%** |   9,860 |
| Running |    79.77% |  34.27% | **47.94%** |   9,978 |
| Jumping |    70.71% |  92.07% | **80.00%** |   8,281 |

### Key Observations

- ✅ **Empty** and **Sitting** achieve near-perfect F1 (>97%) — these stationary/low-motion activities produce consistent Doppler signatures that transfer trivially across environments.
- ✅ **Jumping** generalizes well (F1 = 80.00%) — the broadband Doppler burst signature of jumping is distinctive and environment-independent.
- ⚠️ **Walking** shows moderate performance (F1 = 72.34%) — the periodic gait Doppler pattern learned from only 3 training sessions partially transfers to the Bedroom, but 867 Walking windows are confused with Running due to similar periodic Doppler structures.
- ❌ **Running** has the weakest recall (34.27%) — only 3,419 of 9,978 Running windows are correctly classified. The model confuses 4,285 Running windows with Walking and 2,195 with Jumping. This suggests that the limited training data (3 sessions) does not capture enough Running diversity for cross-environment generalization. 

---

## Confusion Matrix

```
                 Predicted
              Empty  Sitting  Walking  Running  Jumping
  Empty      11718       6        0        0        1
  Sitting        0   11913        0        0        0
  Walking        0       7     8024      867      962
  Running       79       0     4285     3419     2195
  Jumping        5     638       14        0     7624
```

### Confusion Patterns

| Misclassification  | Count | Explanation                                                                                                                                     |
| :----------------- | ----: | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| Running → Walking | 4,285 | Periodic Doppler patterns overlap; Running has slightly higher frequency ridges that the model fails to discriminate with limited training data |
| Running → Jumping | 2,195 | Both produce high-energy broadband Doppler events; transient bursts in Running can mimic short Jump signatures                                  |
| Walking → Running |   867 | Reverse of above — fast Walking and slow Running produce nearly identical Doppler spectra                                                      |
| Walking → Jumping |   962 | Likely due to abrupt direction changes during Walking that create brief broadband bursts                                                        |
| Jumping → Sitting |   638 | Pause periods between jumps produce near-zero Doppler, resembling Sitting                                                                       |

---

## Training History

| Metric                   | Epoch 1 | Epoch 5 |   Best   | Final (Epoch 25) |
| :----------------------- | :------: | :------: | :------: | :--------------: |
| **Train Accuracy** |  97.81%  | 100.00% | 100.00% |     100.00%     |
| **Train Loss**     | 0.00625 | 4.65e-06 | 1.50e-07 |     1.80e-07     |
| **Val Accuracy**   |  99.98%  | 100.00% | 100.00% |     100.00%     |
| **Val Loss**       | 3.60e-05 | 8.46e-08 | 6.16e-10 |     6.16e-10     |

### Learning Rate Schedule

The learning rate was reduced 7 times by `ReduceLROnPlateau` (factor=0.5):

```
Epochs  1–4  :  1.00e-04
Epochs  5–7  :  5.00e-05
Epoch   8–10 :  2.50e-05
Epochs 11–13 :  1.25e-05
Epochs 14–16 :  6.25e-06
Epochs 17–19 :  3.13e-06
Epochs 20–22 :  1.56e-06
Epochs 23–25 :  1.00e-06  (min_lr reached)
```

> The model converges to perfect training and validation accuracy by epoch 4, indicating that the within-environment task (Living Room + Lab → Living Room + Lab) is trivially solvable. The real challenge is the **cross-environment gap** when evaluating on the held-out Bedroom.

---

## Hyperparameters

| Parameter      | Value                                   |
| :------------- | :-------------------------------------- |
| Window Size    | 340 × 100 (time × freq)               |
| Stride         | 17 samples                              |
| Batch Size     | 32                                      |
| Max Epochs     | 25                                      |
| Optimizer      | AdamW                                   |
| Learning Rate  | 1e-4 (initial)                          |
| Weight Decay   | 1e-4                                    |
| Loss Function  | CategoricalFocalCrossentropy (γ = 2.0) |
| Early Stopping | Patience = 10 (monitors`val_loss`)    |
| LR Reduction   | Factor = 0.5, patience = 3 (min: 1e-6)  |
| Checkpointing  | Best`val_accuracy`                    |

### Model Architecture

```
Input (340, 100, 1) — single antenna
  → Stem Conv2D(64, 3×3) + BN + ReLU
  → ResBlock(64)  + MaxPool(2×2)
  → ResBlock(128) + MaxPool(2×2)
  → ResBlock(256)
  → CBAM (Channel + Spatial Attention, r=8, k=7)
  → Reshape → (85, 6400)
  → BiLSTM(128) × 2 (dropout=0.2)
  → GlobalAveragePooling1D
  → Dense(256, ReLU) + Dropout(0.5)
  → Dense(5, Softmax)
```

### Data Augmentation (Training Only)

- **Gaussian Noise**: $\mathcal{N}(0, 0.05)$ added to every window
- **Time Masking**: 30% probability, masks 1–30 contiguous time steps

### Inference Strategy

- **4-Antenna Probability Fusion**: Each antenna processed independently; softmax probabilities averaged
- **Temporal Smoothing**: 11-sample uniform filter along time axis

---

## Output Files

| File               |    Size | Description                                                       |
| :----------------- | ------: | :---------------------------------------------------------------- |
| `model.keras`    | 96.2 MB | Trained model weights (best val_accuracy checkpoint)              |
| `history.json`   |  2.7 KB | Full training curves (loss, accuracy, val_loss, val_accuracy, lr) |
| `norm_stats.pkl` |   976 B | Z-score normalization mean/std computed from training data        |
| `results.pkl`    |   852 B | Per-class metrics, confusion matrix, dataset sizes                |

### Loading Results

```python
import pickle

# Load metrics
with open('fold_1/results.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Macro F1: {results['macro_f1']:.4f}")
print(f"Classes:  {results['classes']}")
print(f"Confusion Matrix:\n{results['conf_matrix']}")
```

```python
# Load trained model
import tensorflow as tf
model = tf.keras.models.load_model('fold_1/model.keras')
model.summary()
```

```python
# Load normalization stats (required for inference)
with open('fold_1/norm_stats.pkl', 'rb') as f:
    norm = pickle.load(f)
mean, std = norm['mean'], norm['std']
# Apply: normalized_window = (raw_window - mean) / std
```

---

## Reproducibility

To reproduce this fold:

```bash
cd SHARP_plus_plus/5class_loeo
python train_loeo.py \
    --doppler_dir ../doppler_traces \
    --window 340 \
    --stride 17 \
    --batch_size 32 \
    --epochs 25 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --dropout 0.5 \
    --lstm_units 128 \
    --dense_units 256 \
    --patience 10 \
    --folds 1
```

Or on **Kaggle**, use the self-contained `kaggle_fold1.py` script.

---

## Interpretation

Fold 1 represents the **most data-starved training scenario** in the LOEO protocol:

- **Only 3 training sessions** (S6a, S6b from Living Room + S7a from Lab) vs. 10 sessions in Fold 2 and 11 sessions in Fold 3.
- Despite this limitation, the model achieves **82.50% accuracy** — driven by near-perfect performance on stationary activities (Empty, Sitting) and strong Jumping recognition.
- The primary failure mode is **Running ↔ Walking confusion**, where the limited training data fails to provide sufficient diversity of gait Doppler signatures across different walking/running speeds and room geometries.
- The **CBAM attention module** is critical here: it suppresses the Bedroom-specific static reflections that dominate the test spectrograms, allowing the model to focus on the Doppler motion signatures learned from the Living Room and Laboratory.
