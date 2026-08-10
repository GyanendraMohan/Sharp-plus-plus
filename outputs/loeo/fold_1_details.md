SHARP++ 5-Class LOEO Experiment

Objective

Evaluate the SHARP++ classifier on the original SHARP five-class activity task using a strict three-fold Leave-One-Environment-Out (LOEO) protocol.

1. Classes

Use only these five classes:

Empty

Sitting

Walking

Running

Jumping

Class mapping:

CLASS_NAMES = [
"Empty",
"Sitting",
"Walking",
"Running",
"Jumping"
]

2. Doppler Input

Use the SHARP-compatible Doppler representation:

340 × 100

For each antenna, the classifier input is:

(340, 100, 1)

Use the existing SHARP preprocessing, phase sanitization, and Doppler extraction pipeline without changing the signal-processing configuration.

Only change the Doppler window to 340 × 100.

3. SHARP++ Architecture

Use the existing SHARP++ architecture:

340 × 100 × 1
↓
Residual CNN
↓
CBAM
↓
Bi-LSTM
↓
Dense
↓
5-class Softmax

Change the final classification layer to:

Dense(5, activation="softmax")

4. Strict 3-Fold LOEO

Fold 1 — Bedroom Held Out

Training: Living Room + Laboratory
Testing: Bedroom

Fold 2 — Living Room Held Out

Training: Bedroom + Laboratory
Testing: Living Room

Fold 3 — Laboratory Held Out

Training: Bedroom + Living Room
Testing: Laboratory

The held-out environment must never be used for:

Training

Validation

Normalization

Model selection

5. Loss Function

Use Focal Loss:

[L=-lpha_y(1-p_y)^\gamma\log(p_y)]

Use:

gamma = 2

Keep the existing SHARP++ optimizer and training configuration unchanged.

6. Four-Antenna Processing

Process each antenna independently:

Antenna 1 → SHARP++ model
Antenna 2 → SHARP++ model
Antenna 3 → SHARP++ model
Antenna 4 → SHARP++ model

For every test sample, obtain the five-class softmax probabilities from each antenna:

P1
P2
P3
P4

Perform probability-level fusion:

[P_{final}(c)=rac{P_1(c)+P_2(c)+P_3(c)+P_4(c)}{4}]

Final prediction:

prediction = argmax(P_final)

7. Temporal Smoothing

Apply the existing 11-sample temporal smoothing used in the current SHARP++ pipeline.

Do not change the smoothing configuration.

8. Final Configuration

Classes : 5
Doppler window : 340 × 100
Single antenna input : 340 × 100 × 1
Architecture : SHARP++ Residual CNN + CBAM + Bi-LSTM
Loss : Focal Loss
Focal gamma : 2
Antennas : 4
Fusion : Probability averaging
Temporal smoothing : 11 samples
Evaluation : Strict 3-fold LOEO

9. Required Results

For each of the three LOEO folds, report:

Accuracy

Macro-F1

Per-class F1

Confusion matrix

Finally report:

Average Accuracy across the three folds

Average Macro-F1 across the three folds

10. Experiment Summary

5 classes

- 340 × 100 Doppler
- SHARP++ Residual CNN + CBAM + Bi-LSTM
- Focal Loss (gamma = 2)
- 4-antenna probability fusion
- 11-sample temporal smoothing
- strict 3-fold LOEO

This is the complete experiment configuration. No additional experiments are required at this stage.

# Complete Details: Fold 1 (SHARP++ LOEO Protocol)

Fold 1 of the Leave-One-Environment-Out (LOEO) evaluation measures how well the model generalizes to the **Bedroom** environment when trained exclusively on data from other rooms.

## Dataset Splitting & Configuration

The protocol enforces strict separation between environments to ensure no data leakage.

### Testing Dataset (Held-out Environment)

- **Environment:** Bedroom
- **Sessions Used:** `S1a`, `S1b`, `S1c`, `S2a`, `S2b`, `S3a`, `S4a`, `S4b`, `S5a`
- **Total Test Samples (Windows):** `223,590`

### Training & Validation Dataset

- **Environments:** Living Room + Laboratory
- **Sessions Used:**
  - _Living Room:_ `S6a`, `S6b`
  - _Laboratory:_ `S7a`
- **Total Training Samples:** `50,795` (75% of the train+val split)
- **Total Validation Samples:** `16,662` (25% of the train+val split)

### Preprocessing Rules Enforced

- **Window Size:** `100` samples per window
- **Overlap/Stride:** `5` samples
- **Normalization:** Z-score normalization was applied. The `mean` and `std` statistics were computed **exclusively** on a random subset (max 5,000 samples) of the 50,795 training windows. No validation or test data influenced the normalization.
- **Input Shape:** `(100, 100, 4)`

---

## Model Architecture

The SHARP++ model uses a **Residual CNN + CBAM Attention + Bi-LSTM** architecture designed to extract spatial-temporal features from WiFi Doppler spectrograms.

1. **Stem (Conv2D):** A 64-filter 3x3 Conv2D layer + BatchNormalization + ReLU.
2. **Residual CNN Backbone:**
   - Stage 1: Residual Block (64 filters) + MaxPooling2D (2x2)
   - Stage 2: Residual Block (128 filters) + MaxPooling2D (2x2)
   - Stage 3: Residual Block (256 filters)
3. **CBAM Attention:** A Convolutional Block Attention Module consisting of Channel Attention (reduction ratio=8) followed by Spatial Attention (7x7 kernel). This suppresses background noise/static reflections and highlights human Doppler motion.
4. **Reshape:** Flattens the frequency and channel dimensions to feed into the temporal layers.
5. **Temporal Modeling (Bi-LSTM):** Two stacked Bidirectional LSTM layers, each with 128 hidden units (256 total per layer), with 20% dropout.
6. **Global Average Pooling (1D):** Aggregates temporal features into a single vector.
7. **Dense Embedding:** A 256-unit Dense layer with ReLU activation and 50% dropout.
8. **Classification Head:** An 8-unit Dense layer with Softmax activation.

---

## Training Methodology

- **Optimizer:** AdamW (Learning Rate = `1e-4`, Weight Decay = `1e-4`)
- **Loss Function:** Sparse Categorical Crossentropy
- **Batch Size:** `32`
- **Max Epochs:** `25`
- **Data Augmentation:** Applied during training.
- **Callbacks:**
  - _Model Checkpoint:_ Saves the model with the best `val_accuracy`.
  - _Early Stopping:_ Monitors `val_loss` with a patience of `10` epochs (restores best weights).
  - _ReduceLROnPlateau:_ Reduces the learning rate by half (factor=0.5) if `val_loss` plateaus for `3` epochs.
- **Decision Fusion (Testing):** During evaluation on the test set, predictions are smoothed across consecutive windows using a uniform 1D filter (size=11) to remove short-term noise.

---

## Overall Performance Metrics

These metrics represent the model's performance on the completely unseen Bedroom test set.

- **Overall Accuracy:** `70.95%`
- **Macro Precision:** `64.21%`
- **Macro Recall:** `65.96%`
- **Macro F1-Score:** `59.67%`

---

## Per-Class Metrics

A detailed breakdown of performance across all 8 activity classes.

| Activity          | Symbol | Precision | Recall | F1-Score | Total Support (N) |
| :---------------- | :----: | :-------- | :----- | :------- | :---------------- |
| **Empty**         |  `E`   | 99.91%    | 99.21% | 99.56%   | 40,286            |
| **Walking**       |  `W`   | 94.48%    | 21.12% | 34.53%   | 33,941            |
| **Running**       |  `R`   | 72.76%    | 76.61% | 74.64%   | 34,349            |
| **Jumping**       |  `J`   | 52.43%    | 93.82% | 67.27%   | 28,950            |
| **Sitting Down**  |  `L`   | 90.36%    | 90.29% | 90.32%   | 40,931            |
| **Standing Up**   |  `S`   | 68.25%    | 60.59% | 64.19%   | 18,896            |
| **Falling**       |  `C`   | 32.04%    | 85.97% | 46.69%   | 11,162            |
| **Other/Unknown** |  `H`   | 3.45%     | 0.09%  | 0.18%    | 15,075            |

---

## Confusion Matrix

The confusion matrix shows the predicted vs. actual classifications.
_Rows represent the **Actual** (True) classes._
_Columns represent the **Predicted** classes._

| Actual \ Predicted | E          | W         | R          | J          | L          | S          | C         | H      |
| :----------------- | :--------- | :-------- | :--------- | :--------- | :--------- | :--------- | :-------- | :----- |
| **E (Empty)**      | **39,967** | 2         | 0          | 0          | 317        | 0          | 0         | 0      |
| **W (Walking)**    | 3          | **7,169** | 9,767      | 17,002     | 0          | 0          | 0         | 0      |
| **R (Running)**    | 0          | 414       | **26,316** | 7,619      | 0          | 0          | 0         | 0      |
| **J (Jumping)**    | 0          | 3         | 0          | **27,160** | 0          | 2          | 1,607     | 178    |
| **L (Sitting)**    | 33         | 0         | 86         | 23         | **36,955** | 3,788      | 0         | 46     |
| **S (Standing)**   | 0          | 0         | 0          | 0          | 3,608      | **11,449** | 3,682     | 157    |
| **C (Falling)**    | 0          | 0         | 0          | 0          | 19         | 1,536      | **9,596** | 11     |
| **H (Other)**      | 0          | 0         | 0          | 0          | 0          | 0          | 15,061    | **14** |
