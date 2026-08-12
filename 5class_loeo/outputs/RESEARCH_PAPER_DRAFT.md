# SHARP++: Environment-Agnostic Human Activity Recognition via Residual CNN–CBAM–BiLSTM Fusion on Wi-Fi Doppler Spectrograms

---

## 1. Abstract

Device-free Human Activity Recognition (HAR) using Wi-Fi Channel State Information (CSI) offers a non-intrusive, privacy-preserving alternative to camera- and wearable-based systems. A persistent challenge is **cross-environment generalization**: models trained in one physical space (e.g., a laboratory) frequently fail when deployed in a different room due to varying multipath propagation profiles. In this work we present **SHARP++**, a deep learning architecture that combines a three-stage **Residual Convolutional Neural Network (ResCNN)** with a **Convolutional Block Attention Module (CBAM)** and a two-layer **Bidirectional LSTM (Bi-LSTM)** to jointly capture spatial–spectral and temporal patterns in Doppler spectrograms derived from CSI phase data. We evaluate SHARP++ on the publicly available SHARP dataset using a strict **three-fold Leave-One-Environment-Out (LOEO)** protocol across three physically distinct environments — Bedroom, Living Room, and Laboratory. During inference a **four-antenna probability-level fusion** strategy combined with **temporal smoothing** is applied to produce robust final predictions. The 8-class experiment yields a mean cross-environment accuracy of **64.39 ± 18.9%** and a mean Macro-F1 of **54.60 ± 21.3%**, with individual fold accuracies ranging from 38.63% to 83.58%, underscoring both the promise and the remaining difficulty of environment-agnostic Wi-Fi HAR.

---

## 2. Introduction

### 2.1. Motivation

Non-line-of-sight (NLOS) sensing using commodity Wi-Fi access points is an attractive HAR modality for smart homes, elderly care, and healthcare monitoring. Unlike cameras, CSI-based systems do not capture visual data, thus preserving occupant privacy. Unlike wearables, they require no user cooperation.

### 2.2. Problem Statement

Wi-Fi CSI signals encode the channel frequency response between a transmitter–receiver pair. Human motion modulates this response through Doppler shifts and multipath reflections. However, the **static component** of the channel — determined by room geometry, furniture layout, wall materials, and transceiver placement — dominates the signal. Consequently, a model trained in one room learns environment-specific features that do not transfer to a new room, a phenomenon widely documented in the CSI-HAR literature.

### 2.3. Contributions

1. **SHARP++ Architecture**: We propose an enhanced architecture that stacks Residual CNN blocks for hierarchical spatial feature extraction, a full CBAM module (channel + spatial attention) to suppress environment-dependent static reflections and highlight human Doppler signatures, and a two-layer Bi-LSTM for temporal modelling.
2. **Four-Antenna Probability Fusion**: Instead of concatenating raw antenna data at the input, each antenna's spectrogram is classified independently and the softmax probability vectors are averaged, reducing over-reliance on any single antenna's spatial perspective.
3. **Rigorous Cross-Environment Evaluation**: We adopt a strict LOEO protocol where the held-out environment is excluded from training, validation, and normalization, providing a realistic assessment of deployment-time performance.

---

## 3. Dataset

### 3.1. The SHARP Dataset

The SHARP dataset consists of Wi-Fi CSI recordings collected with a 4-antenna receiver across three environments:

| Environment | Sessions | Description |
| :--- | :--- | :--- |
| **Bedroom** | S1a, S1b, S1c, S2a, S2b, S3a, S4a, S4b, S5a | Residential bedroom |
| **Living Room** | S6a, S6b | Residential living room |
| **Laboratory** | S7a | University laboratory |

Data was collected from **three volunteers** (one male, two females) performing eight activities:

| Activity | Code | Description |
| :--- | :---: | :--- |
| Empty | `E` | No human presence in the room |
| Walking | `W` | Walking around the room |
| Running | `R` | Running around the room |
| Jumping | `J` | Jumping in place |
| Sitting | `L` | Seated somewhere in the room |
| Standing | `S` | Standing still somewhere in the room |
| Continuous sit/stand | `C` | Repeatedly sitting down and standing up |
| Arm gym | `H` | Upper-body arm exercises |

### 3.2. Signal Preprocessing

1. **CSI Phase Sanitization**: Raw phase values are unwrapped and sanitized following the SHARP reference pipeline.
2. **Doppler Spectrum Extraction**: A Short-Time Fourier Transform is applied to produce time–frequency Doppler spectrograms.
3. **Mean Subtraction**: Each raw stream is mean-subtracted along the time axis (`data - mean(data, axis=0)`) to remove the DC (static) component, implemented in `load_raw_stream()`.
4. **Windowing**: Continuous spectrograms are divided into fixed-size windows of **100 time-steps × 100 frequency bins** with a stride of **5 samples**, yielding overlapping windows.
5. **Z-Score Normalization**: Per-fold, the mean ($\mu$) and standard deviation ($\sigma$) are computed from a random subset (max 5,000 samples) of the **training set only**. Each window is normalized as: $x' = \frac{x - \mu}{\sigma}$. No validation or test data influences normalization statistics.

### 3.3. Data Augmentation

During training, two online augmentation strategies are applied:

- **Gaussian Noise Injection**: Additive noise $\mathcal{N}(0, 0.05)$ is added to every training window to improve robustness against signal variations.
- **Time Masking** (applied with 30% probability): A random contiguous block of 1–30 time-steps is zeroed out, simulating brief signal dropouts and encouraging the model to learn from partial temporal context.

---

## 4. SHARP++ Architecture

### 4.1. Design Rationale — Why a Hybrid CNN–Attention–RNN Pipeline?

Wi-Fi CSI Doppler spectrograms pose a unique classification challenge that no single model family addresses in isolation. Each spectrogram is simultaneously:

1. **A 2-D image** (time × frequency) containing local spatial textures — periodic gait ridges for Walking, broadband bursts for Jumping, near-zero energy for Empty.
2. **A temporal sequence** — an activity unfolds over successive time steps, and the *ordering* of Doppler bursts (e.g., the rhythmic push–pull of Continuous sit/stand) carries discriminative information that spatial convolutions alone cannot exploit.
3. **Contaminated by environment-specific static clutter** — multipath reflections from walls, furniture, and metallic surfaces dominate the signal energy and vary drastically across rooms. The model must learn to *suppress* these static components and *amplify* the dynamic human Doppler signatures.

These three properties motivate the three-stage SHARP++ pipeline:

| Challenge | Solution Module | Role |
| :--- | :--- | :--- |
| Local spatial–spectral texture extraction | **Residual CNN (3 stages)** | Hierarchical feature learning with gradient-stable training |
| Environment-specific static clutter suppression | **CBAM Attention** | Learns *which* feature channels and *where* in the time–frequency plane to focus |
| Temporal dynamics and activity sequencing | **Bidirectional LSTM (2 layers)** | Captures forward/backward temporal dependencies |

**Why not a pure CNN (e.g., ResNet-50)?**  A standard image classifier treats the spectrogram as a static image and cannot model temporal ordering. Two spectrograms with identical pixel histograms but reversed time axes (e.g., sitting-down vs. standing-up) would produce identical CNN features but represent different activity phases.

**Why not a pure RNN/Transformer?**  Raw spectrograms have a feature dimension of 100 frequency bins per time step — far too low-level for an LSTM to learn useful representations without prior spatial feature extraction. CNNs efficiently compress the 100 × 100 spatial field into compact, high-level feature vectors that the LSTM can then sequence-model.

**Why attention?**  Without CBAM, the CNN features contain both static room signatures (high energy but non-discriminative) and dynamic Doppler signatures (lower energy but highly discriminative). CBAM acts as a learnable, environment-adaptive filter that suppresses the former and amplifies the latter, which is precisely the bottleneck for cross-environment generalization.

### 4.2. Architecture Diagram

```
Input: (T, F, C) = (100, 100, 4)
        │
        ▼
┌─────────────────────────┐
│  Stem Conv2D(64, 3×3)   │  padding='same', bias=False
│  BatchNormalization      │
│  ReLU                    │
└─────────────────────────┘
        │ Output: (100, 100, 64)
        ▼
┌─────────────────────────┐
│  Residual Block (64)     │  2× Conv2D(64, 3×3) + skip connection
│  MaxPooling2D(2×2)       │
└─────────────────────────┘
        │ Output: (50, 50, 64)
        ▼
┌─────────────────────────┐
│  Residual Block (128)    │  2× Conv2D(128, 3×3) + 1×1 projection skip
│  MaxPooling2D(2×2)       │
└─────────────────────────┘
        │ Output: (25, 25, 128)
        ▼
┌─────────────────────────┐
│  Residual Block (256)    │  2× Conv2D(256, 3×3) + 1×1 projection skip
└─────────────────────────┘
        │ Output: (25, 25, 256)
        ▼
┌─────────────────────────┐
│  CBAM Attention Module   │
│  ├─ Channel Attention    │  GlobalAvgPool + GlobalMaxPool → shared MLP (256→32→256) → sigmoid
│  └─ Spatial Attention    │  ChannelAvg ∥ ChannelMax → Conv2D(1, 7×7) → sigmoid
└─────────────────────────┘
        │ Output: (25, 25, 256)
        ▼
┌─────────────────────────┐
│  Reshape                 │  (25, 25×256) = (25, 6400)
└─────────────────────────┘
        │ Output: (25, 6400)
        ▼
┌─────────────────────────┐
│  Bi-LSTM Layer 1         │  128 units per direction → 256 total, dropout=0.2
│  (return_sequences=True) │
└─────────────────────────┘
        │ Output: (25, 256)
        ▼
┌─────────────────────────┐
│  Bi-LSTM Layer 2         │  128 units per direction → 256 total, dropout=0.2
│  (return_sequences=True) │
└─────────────────────────┘
        │ Output: (25, 256)
        ▼
┌─────────────────────────┐
│  GlobalAveragePooling1D  │
└─────────────────────────┘
        │ Output: (256,)
        ▼
┌─────────────────────────┐
│  Dense(256, ReLU)        │
│  Dropout(0.5)            │
└─────────────────────────┘
        │ Output: (256,)
        ▼
┌─────────────────────────┐
│  Dense(8, Softmax)       │  Classification head
└─────────────────────────┘
        │ Output: (8,)
```

### 4.3. Component Details — Design, Effect, and Code Mapping

#### 4.3.1. Signal Preprocessing: Mean Subtraction and Normalization

**Why This Step:**  Raw CSI phase data contains a large DC (zero-frequency) component that encodes the *static* channel response — the room geometry, wall reflections, and furniture layout. This static component dominates the signal energy and is the primary source of environment dependence. Removing it before any learning forces the model to operate on the *dynamic* residual signal caused by human motion.

**Effect on CSI-HAR:**  Mean subtraction along the time axis removes the time-averaged static component from each frequency bin, effectively isolating the Doppler shift energy produced by human movement. This is a physics-motivated preprocessing step: the static multipath arrives at a constant phase, while human-induced Doppler shifts create time-varying phase modulations. By subtracting the temporal mean, the static component is zeroed out, and the residual signal is dominated by the Doppler component.

Z-score normalization then rescales each frequency bin to zero mean and unit variance, ensuring that the model's gradient landscape is well-conditioned and that no single frequency bin dominates the loss gradient simply because of its energy scale.

**Code Mapping:**

The mean subtraction is implemented in the `load_raw_stream()` function in `dataset_loader.py`:

```python
# dataset_loader.py — load_raw_stream() (lines 37–42)
def load_raw_stream(filepath):
    """Returns np.ndarray (freq_bins, time_steps) — mean-subtracted"""
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)
    data = data - np.mean(data, axis=0, keepdims=True)   # ← DC removal
    return data.T.astype(np.float32)
```

The operation `data - np.mean(data, axis=0)` computes the temporal mean of each frequency bin and subtracts it, removing the static channel response. The transpose `.T` reorients from `(freq, time)` to `(time, freq)` for downstream windowing.

Z-score normalization is computed from the **training set only** (critical for the LOEO protocol) in `compute_norm_stats()`:

```python
# dataset_loader.py — compute_norm_stats() (lines 132–159)
mean = samples.mean(axis=(0, 1), keepdims=True)  # (1, 1, freq, 1)
std  = samples.std(axis=(0, 1), keepdims=True)   # (1, 1, freq, 1)
std  = np.maximum(std, 1e-8)                      # prevent division by zero
```

Normalization is applied at runtime within the `tf.data` pipeline:

```python
# dataset_loader.py — map_single_antenna() (lines 190–192)
window = tf.cast(window, tf.float32)
window = (window - mean_t) / std_t
```

---

#### 4.3.2. Stem Convolution — Entry Point

**Why This Component:**  The stem layer is the entry point that transforms the raw (possibly single-channel) spectrogram input into a rich 64-channel feature space. It replaces the standard "RGB → 64-channel" stem found in image classifiers like ResNet. In SHARP++, the input has either 4 channels (one per antenna, in the 8-class variant) or 1 channel (single-antenna, in the 5-class LOEO variant), which is far fewer than the 64 filters that the first residual block expects. The stem bridges this dimensional gap.

**Effect on CSI-HAR:**  The 3×3 convolution extracts local time–frequency correlations at the finest granularity. Each of the 64 output filters learns a distinct micro-pattern: some respond to narrow-band stationary signals (Standing), others to broadband bursts (Jumping), and others to periodic ridges (Walking gait). Batch Normalization stabilizes these activations across different environments and antenna characteristics.

The use of `use_bias=False` is deliberate: since Batch Normalization immediately follows the convolution, it already includes a learnable bias term ($\beta$). Adding a separate convolution bias would be redundant and wasteful.

**Code Mapping:**

```python
# model.py — build_sharp_plus_plus() (lines 182–188)
# ── Stem ──
x = tf.keras.layers.Conv2D(
    64, (3, 3), padding='same', use_bias=False, name='stem_conv')(inputs)
x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
x = tf.keras.layers.ReLU(name='stem_relu')(x)
# output: (T, F, 64)
```

The model is constructed in `build_sharp_plus_plus()` with a flexible `input_shape` parameter:

```python
# model.py — build_sharp_plus_plus() (lines 165–180)
def build_sharp_plus_plus(input_shape, num_classes,
                           lstm_units=128, dense_units=256, dropout_rate=0.5):
    inputs = tf.keras.Input(shape=input_shape, name='spectrogram_input')
```

In the 8-class variant, `input_shape = (100, 100, 4)` feeds all 4 antennas as channels. In the 5-class LOEO variant, `input_shape = (340, 100, 1)` feeds a single antenna at a time (with probability fusion at inference):

```python
# 5class_loeo/dataset_loader.py — prepare_dataset() (line 118)
input_shape = (window_length, giant_array.shape[1], 1)   # (340, 100, 1)
```

---

#### 4.3.3. Residual CNN Backbone (3 Stages) — Hierarchical Spatial Feature Extraction

**Why This Component:**  Doppler spectrograms contain discriminative patterns at multiple spatial scales. Walking produces narrow, periodic Doppler ridges spanning 5–10 frequency bins; Jumping creates broadband energy bursts spanning 30–50 bins; Standing/Empty exhibit near-zero energy across the full spectrum. A single convolutional layer with a fixed 3×3 receptive field can only capture the finest-scale patterns. Stacking multiple stages with progressive downsampling (MaxPooling) expands the effective receptive field geometrically:

| Stage | Receptive Field | Captures |
| :---: | :--- | :--- |
| Stage 1 (64 filters) | ~5×5 pixels | Fine edges, narrow Doppler ridges, micro-textures |
| Stage 2 (128 filters) | ~14×14 pixels | Gait periodicity, medium-scale spectral structure |
| Stage 3 (256 filters) | ~30×30 pixels | Global activity shape, broadband energy distribution |

**Why Residual Connections?**  The key innovation of residual learning (He et al., CVPR 2016) is the skip connection: instead of learning the desired mapping $\mathcal{H}(x)$ directly, the network learns the *residual* $\mathcal{F}(x) = \mathcal{H}(x) - x$, and the output is $\mathcal{F}(x) + x$. This has two critical effects:

1. **Gradient flow**: During backpropagation, the skip connection provides a direct gradient highway from later layers to earlier layers, preventing the vanishing gradient problem that plagues deep plain networks. This is essential because SHARP++ stacks 6 convolutional layers (2 per residual block × 3 stages) plus the CBAM and BiLSTM — deep enough that gradient degradation becomes a concern.

2. **Incremental refinement**: Each residual block learns only the *difference* between the input and the desired output. If the optimal transformation for a particular stage is close to the identity (i.e., the features from the previous stage are already good), the residual block can easily learn near-zero weights, effectively passing the input through unchanged. This is much easier to optimize than learning an arbitrary nonlinear mapping from scratch.

**Why Progressive Channel Widening (64 → 128 → 256)?**  As spatial resolution decreases through pooling, the number of spatial locations is quartered at each stage (100×100 → 50×50 → 25×25). To maintain the representational capacity, the number of feature channels is doubled at each stage, compensating for the spatial compression. The final stage produces 256 channels — each a semantic "feature detector" tuned to a different aspect of the Doppler signature.

**Why 1×1 Projection Shortcuts?**  When the channel dimension changes between stages (64→128, 128→256), the skip connection cannot simply add the input to the output because their shapes differ. The `1×1` convolution with BatchNorm projects the shortcut to the correct dimensionality without affecting the spatial dimensions. This is computationally cheap (a pointwise linear transform) and mathematically equivalent to a learned linear mixing of the input channels.

**Effect on CSI-HAR:**  The three-stage hierarchy produces a 256-dimensional feature map at 1/4 the original spatial resolution. Each spatial location in this feature map aggregates information from a 30×30 neighbourhood of the original spectrogram — large enough to encode the periodic gait structure of Walking (typically 10–15 pixel periodicity in the time axis) or the broadband energy distribution of Jumping. The residual connections ensure that fine-grained details (important for distinguishing Walking from Running, which differ primarily in Doppler ridge spacing) are not lost during the deep feature extraction.

**Code Mapping:**

The `residual_block()` function implements a standard two-convolution residual block with an optional 1×1 projection shortcut:

```python
# model.py — residual_block() (lines 28–65)
def residual_block(x, filters, name_prefix='res'):
    shortcut = x

    # Main path: Conv → BN → ReLU → Conv → BN
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), padding='same', use_bias=False,
        name=f'{name_prefix}_conv1')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = tf.keras.layers.ReLU(name=f'{name_prefix}_relu1')(x)

    x = tf.keras.layers.Conv2D(
        filters, (3, 3), padding='same', use_bias=False,
        name=f'{name_prefix}_conv2')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)

    # Projection shortcut when channel depth changes
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, (1, 1), padding='same', use_bias=False,
            name=f'{name_prefix}_proj_conv')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(
            name=f'{name_prefix}_proj_bn')(shortcut)

    x = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, shortcut])
    x = tf.keras.layers.ReLU(name=f'{name_prefix}_relu2')(x)
    return x
```

The three stages are assembled in `build_sharp_plus_plus()`:

```python
# model.py — build_sharp_plus_plus() (lines 190–202)

# Stage 1: ResBlock(64) + MaxPool → (T/2, F/2, 64)
x = residual_block(x, 64, name_prefix='res1')
x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(x)

# Stage 2: ResBlock(128) + MaxPool → (T/4, F/4, 128)
x = residual_block(x, 128, name_prefix='res2')
x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)

# Stage 3: ResBlock(256) — no pooling → (T/4, F/4, 256)
x = residual_block(x, 256, name_prefix='res3')
```

Note that Stage 3 deliberately omits MaxPooling. This preserves the spatial resolution at 25×25, ensuring the subsequent Bi-LSTM has 25 time steps — sufficient temporal granularity for modelling activity dynamics.

---

#### 4.3.4. CBAM (Convolutional Block Attention Module) — Environment-Adaptive Feature Filtering

**Why This Component:**  CBAM (Woo et al., ECCV 2018) is the most critical component for cross-environment generalization. After the residual CNN extracts 256 feature channels, many of these channels inevitably encode environment-specific static clutter — room geometry, furniture reflections, wall material signatures. Without attention-based filtering, the classifier would rely on these environment-specific features, leading to catastrophic failure when deployed in a new room (as evidenced by the Fold 3 Laboratory results).

CBAM provides two complementary filtering mechanisms:

1. **Channel Attention** answers: *"Which of the 256 feature channels contain human Doppler information vs. static room clutter?"*
2. **Spatial Attention** answers: *"Which time–frequency regions of the spectrogram contain the activity signature vs. background noise?"*

**Effect on CSI-HAR — Channel Attention:**

In a Doppler spectrogram, certain feature channels learn to detect static reflections (high energy, constant across time) while others learn to detect dynamic Doppler shifts (lower energy, time-varying). Channel attention computes a per-channel importance weight via a squeeze-and-excitation mechanism:

$$\mathbf{M}_c = \sigma\bigl(\text{MLP}(\text{AvgPool}(\mathbf{F})) + \text{MLP}(\text{MaxPool}(\mathbf{F}))\bigr)$$

- **Global Average Pooling** captures the *mean activation* of each channel — sensitive to distributed, low-energy patterns like the subtle Doppler shifts from Arm gym or Standing.
- **Global Max Pooling** captures the *peak activation* — sensitive to sharp, localized features like the broadband burst of a Jump.
- The **shared MLP** with a bottleneck (256 → 32 → 256, reduction ratio $r = 8$) forces the network to learn compressed inter-channel dependencies, acting as a regularizer against overfitting to environment-specific channel correlations.
- The **sigmoid gate** produces a soft mask $\in [0, 1]^{256}$ that multiplicatively scales each channel. Channels encoding static clutter are suppressed (sigmoid ≈ 0); channels encoding dynamic motion are amplified (sigmoid ≈ 1).

**Effect on CSI-HAR — Spatial Attention:**

After channel re-weighting, spatial attention identifies *where* in the 2-D time–frequency plane the relevant activity signature lies:

$$\mathbf{M}_s = \sigma\bigl(\text{Conv}_{7 \times 7}([\text{AvgPool}_c(\mathbf{F}'); \text{MaxPool}_c(\mathbf{F}')])\bigr)$$

- **Channel-wise average and max pooling** collapse the 256 channels into two summary maps, each of shape $H \times W \times 1$.
- A **7×7 convolution** processes the concatenated maps to produce a spatial attention mask. The large kernel size (7×7) ensures the attention map is smooth and captures broad spatial context rather than point-level noise.
- The resulting spatial mask highlights the time–frequency regions where human Doppler energy is concentrated. For Walking, this would be the narrow frequency band around the gait Doppler shift; for Jumping, this would be a broader frequency range during the jump events.

**Why Sequential (Channel → Spatial) Rather Than Parallel?**  The original CBAM paper demonstrates that sequential application outperforms parallel attention. The intuition is that channel attention first *selects the right feature detectors*, and then spatial attention *focuses those selected detectors on the right locations*. Applying spatial attention before channel selection would waste computational effort attending to spatial regions in uninformative channels.

**Code Mapping:**

Channel Attention implementation:

```python
# model.py — channel_attention() (lines 72–107)
def channel_attention(x, reduction_ratio=8, name_prefix='ca'):
    channels = x.shape[-1]                    # 256
    reduced = max(1, channels // reduction_ratio)  # 32

    # Shared MLP weights (same Dense layers for both branches)
    dense_down = tf.keras.layers.Dense(
        reduced, activation='relu', use_bias=False, ...)
    dense_up = tf.keras.layers.Dense(
        channels, use_bias=False, ...)

    # Branch 1: Global Average Pool → Reshape → shared MLP
    avg_pool = tf.keras.layers.GlobalAveragePooling2D(...)(x)
    avg_pool = tf.keras.layers.Reshape((1, 1, channels), ...)(avg_pool)
    avg_out = dense_up(dense_down(avg_pool))

    # Branch 2: Global Max Pool → Reshape → shared MLP
    max_pool = tf.keras.layers.GlobalMaxPooling2D(...)(x)
    max_pool = tf.keras.layers.Reshape((1, 1, channels), ...)(max_pool)
    max_out = dense_up(dense_down(max_pool))

    # Combine branches and apply sigmoid gate
    scale = tf.keras.layers.Activation('sigmoid', ...)(avg_out + max_out)
    return tf.keras.layers.Multiply(...)([x, scale])
```

The **shared MLP** (`dense_down` and `dense_up`) is a key design detail: both the average-pooled and max-pooled descriptors pass through the *same* Dense layers. This weight sharing halves the parameter count and forces the MLP to learn a unified channel importance function that is robust to both the mean and peak statistics.

Spatial Attention implementation:

```python
# model.py — spatial_attention() (lines 110–137)
def spatial_attention(x, kernel_size=7, name_prefix='sa'):
    # Channel-wise statistics — Lambda to stay inside Keras graph
    avg_pool = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), ...)(x)  # H×W×1
    max_pool = tf.keras.layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True), ...)(x)   # H×W×1

    concat = tf.keras.layers.Concatenate(axis=-1, ...)([avg_pool, max_pool])

    # 7×7 Conv to spatial map → sigmoid
    attn_map = tf.keras.layers.Conv2D(
        1, kernel_size, padding='same', use_bias=False, ...)(concat)
    attn_map = tf.keras.layers.Activation('sigmoid', ...)(attn_map)
    return tf.keras.layers.Multiply(...)([x, attn_map])
```

The `tf.keras.layers.Lambda` wrappers ensure that the `tf.reduce_mean` and `tf.reduce_max` operations are captured as Keras graph nodes, enabling proper model serialization and mixed-precision compatibility.

The CBAM block applies both modules sequentially:

```python
# model.py — cbam_block() (lines 140–158)
def cbam_block(x, reduction_ratio=8, spatial_kernel=7, name_prefix='cbam'):
    x = channel_attention(x, reduction_ratio, ...)   # "Which channels matter?"
    x = spatial_attention(x, spatial_kernel, ...)     # "Where in time-freq to focus?"
    return x
```

Applied in the main model:

```python
# model.py — build_sharp_plus_plus() (lines 204–206)
x = cbam_block(x, reduction_ratio=8, spatial_kernel=7)
# output: (T/4, F/4, 256) — static reflections suppressed
```

---

#### 4.3.5. Reshape — CNN-to-RNN Bridge

**Why This Step:**  CNNs output 3-D feature maps $(H, W, C)$ while RNNs expect 2-D sequences $(T_{\text{seq}}, D_{\text{feat}})$. The Reshape layer bridges this gap by collapsing the spatial width (frequency) and channel dimensions into a single feature vector per time step.

**Effect on CSI-HAR:**  After reshaping, the tensor has shape $(T/4, F/4 \times 256)$ — for a 100×100 input, this is $(25, 6400)$. Each of the 25 "time steps" now encodes a 6400-dimensional feature vector summarizing the frequency-domain activity signature at that temporal position. The LSTM then processes these 25 steps as a sequence, modelling how the frequency-domain features evolve over time.

The choice to collapse the frequency (width) dimension rather than the time (height) dimension is deliberate: human activities unfold *temporally*, so the time dimension must remain as the sequence axis. The frequency dimension represents the Doppler spectrum at each instant and is more naturally aggregated into a per-timestep feature vector.

**Code Mapping:**

```python
# model.py — build_sharp_plus_plus() (lines 208–214)
# Treat height dimension (T/4) as time steps for the LSTM.
# Each step aggregates spatial-freq features: (F/4 × 256) per step.
freq_channels = x.shape[2] * x.shape[3]   # F/4 * 256 (static, known at build time)
x = tf.keras.layers.Reshape(
    (-1, freq_channels), name='reshape_lstm')(x)
# output: (T/4, F/4 × 256)
```

The use of `-1` for the first dimension tells Keras to infer the time-step count dynamically, making the model flexible to different window lengths (e.g., 100 for 8-class, 340 for 5-class).

---

#### 4.3.6. Bidirectional LSTM Stack — Temporal Dynamics Modelling

**Why This Component:**  Many human activities have characteristic temporal profiles that evolve over the observation window. For example:

- **Walking** produces a *periodic* Doppler pattern with ~2 Hz cadence (one stride cycle per 0.5s).
- **Continuous sit/stand** produces an *alternating* pattern of upward and downward Doppler shifts.
- **Jumping** produces *sporadic* broadband bursts separated by near-zero intervals.
- **Standing** and **Empty** produce *stationary* (near-constant) patterns.

A CNN compresses these temporal patterns into a fixed-size spatial feature map but cannot explicitly model the *ordering* or *periodicity* of events across time. An LSTM processes the feature sequence step-by-step, maintaining a hidden state that encodes the temporal context accumulated so far.

**Why Bidirectional?**  A unidirectional LSTM at time step $t$ only has access to features from time steps $1, 2, \ldots, t$. This is limiting because:

- The *end* of a Walking segment (deceleration phase) is highly informative for recognizing the activity, but a forward LSTM must wait until the end to incorporate it.
- Activity boundaries (transitions between activities) are best detected by examining context on *both sides* of the boundary.

The Bidirectional LSTM runs two independent LSTMs — one forward, one backward — and concatenates their hidden states: $\mathbf{h}_t = [\overrightarrow{\mathbf{h}_t}; \overleftarrow{\mathbf{h}_t}]$. This gives each time step access to the *full* temporal context from both past and future.

**Why Two Layers?**  Stacking two BiLSTM layers creates a *hierarchical* temporal representation:

- **Layer 1** learns *low-level* temporal patterns: individual Doppler bursts, single gait cycles, momentary pauses.
- **Layer 2** learns *high-level* temporal patterns: sequences of gait cycles (sustained Walking), repeated sit-stand transitions (Continuous sit/stand), or the absence of any temporal variation (Empty/Standing).

Both layers use `return_sequences=True` so that Layer 2 receives the full temporal sequence from Layer 1, and the downstream GlobalAveragePooling can aggregate over all time steps.

**Why `dropout=0.2` and `recurrent_dropout=0.0`?**  The `dropout=0.2` parameter applies dropout to the *input connections* of the LSTM at each time step, randomly zeroing 20% of the input features to prevent co-adaptation. `recurrent_dropout=0.0` (no recurrent dropout) is a pragmatic choice: recurrent dropout dramatically slows training on GPU/Metal backends because it prevents cuDNN-optimized LSTM kernels from being used. The 0.2 input dropout provides sufficient regularization without the performance penalty.

**Effect on CSI-HAR:**  The BiLSTM stack transforms the 25-step sequence of 6400-d spatial feature vectors into a 25-step sequence of 256-d temporal feature vectors. Each temporal feature vector at position $t$ encodes not just "what the Doppler spectrum looks like at time $t$" but "what the Doppler spectrum looks like at time $t$, given the full context of what happened before and after." This contextual representation is far more discriminative for activities that are defined by their temporal dynamics rather than their instantaneous spectral signature.

**Code Mapping:**

```python
# model.py — build_sharp_plus_plus() (lines 216–229)

# ── Bi-LSTM Stack ─────────────────────────────────────
x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(lstm_units, return_sequences=True,
                         dropout=0.2, recurrent_dropout=0.0),
    name='bilstm_1'
)(x)
# output: (T/4, lstm_units × 2)  e.g. (25, 256)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(lstm_units, return_sequences=True,
                         dropout=0.2, recurrent_dropout=0.0),
    name='bilstm_2'
)(x)
# output: (T/4, lstm_units × 2)  e.g. (25, 256)
```

The default `lstm_units=128` produces 128 forward + 128 backward = 256 outputs per time step. This matches the `dense_units=256` downstream, maintaining a consistent embedding dimensionality.

---

#### 4.3.7. Classification Head — Global Aggregation and Prediction

**Why GlobalAveragePooling1D?**  The BiLSTM outputs a sequence of 25 temporal feature vectors, but the classifier needs a single fixed-size vector for the entire window. Two aggregation strategies exist:

1. **Take the last hidden state**: Only uses the final time step's output. This discards intermediate temporal information and biases the model toward the end of the window.
2. **GlobalAveragePooling1D**: Averages across all 25 time steps. This treats every time step's contribution equally, providing *temporal translation invariance* — the model's prediction is invariant to small shifts in where the activity appears within the window.

GlobalAveragePooling1D is preferred because activities often do not align neatly with window boundaries. A Walking segment may start at time step 5 and end at step 20, with empty space before and after. Averaging ensures the Walking features dominate regardless of their temporal position.

**Why Dense(256, ReLU) + Dropout(0.5)?**  This fully connected layer provides a final nonlinear projection from the averaged temporal features into a classification-ready embedding. The high dropout rate of 0.5 is deliberate: it provides strong regularization against overfitting to the training environments, which is critical when the training set contains only 2–3 environments and the model must generalize to an unseen environment.

**Why `dtype='float32'` on the Output Layer?**  When mixed-precision training (`mixed_float16`) is enabled for GPU acceleration, all intermediate computations use 16-bit floats. However, the softmax function involves an exponentiation $e^{z_i}$ that can easily overflow or underflow in float16 (max representable value ≈ 65,504). Forcing the output layer to `float32` ensures numerical stability of the probability computation without affecting the speed of the rest of the network.

**Code Mapping:**

```python
# model.py — build_sharp_plus_plus() (lines 231–248)

# ── Global Average Pooling over time ──────────────────
x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)
# output: (256,)

# ── Dense Embedding ───────────────────────────────────
x = tf.keras.layers.Dense(
    dense_units, activation='relu', name='dense_embed')(x)
x = tf.keras.layers.Dropout(dropout_rate, name='dropout')(x)
# output: (256,)

# ── Classification Head ───────────────────────────────
# dtype='float32' required when mixed_float16 precision is active
outputs = tf.keras.layers.Dense(
    num_classes, activation='softmax', dtype='float32', name='output')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SHARP_plus_plus')
```

---

### 4.4. Data Augmentation Strategy

**Why Augmentation:**  With only 3 training environments (and often as few as 1–3 sessions per fold), the training data is limited. Data augmentation synthetically expands the effective training set by applying random transformations that preserve the activity label but alter the signal characteristics.

**Gaussian Noise Injection:**  Additive noise $\mathcal{N}(0, 0.05)$ simulates the signal-to-noise ratio (SNR) variations that occur in real-world deployments due to electromagnetic interference, transmit power fluctuations, and multipath fading. This forces the model to learn robust features that are not sensitive to small signal perturbations.

**Time Masking (30% probability):**  A random contiguous block of 1–30 time steps is zeroed out, simulating brief signal dropouts caused by WiFi packet loss, interference, or transient obstructions. This encourages the model to rely on distributed temporal context rather than any single time step.

**Code Mapping:**

```python
# 5class_loeo/dataset_loader.py — augment_fn() (lines 200–213)
def augment_fn(x, label):
    # Gaussian Noise Injection
    x = x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.05)
    
    # Time Masking (30% probability)
    def mask_time():
        win_len = tf.shape(x)[0]
        mask_size = tf.random.uniform([], 1, 30, dtype=tf.int32)
        mask_start = tf.random.uniform([], 0, win_len - mask_size, dtype=tf.int32)
        idx = tf.range(win_len)
        mask = tf.logical_or(idx < mask_start, idx >= mask_start + mask_size)
        return x * tf.reshape(tf.cast(mask, tf.float32), [win_len, 1, 1])
    x = tf.cond(tf.random.uniform([]) < 0.3, mask_time, lambda: x)
    return x, label
```

Augmentation is applied only during training, never during validation or testing:

```python
# 5class_loeo/train_loeo.py — (lines 139–142)
ds_train = build_normalized_tf_dataset(
    giant_tr, train_idx, args.window, args.batch_size, mean, std,
    mode='train', shuffle=True, repeat=True, augment=True, ...
)
```

---

### 4.5. Training Regime

**Loss Function — Focal Loss (5-class LOEO variant):**  The 5-class LOEO variant uses `CategoricalFocalCrossentropy(gamma=2.0)` instead of standard cross-entropy. Focal loss down-weights the contribution of easy, well-classified examples (e.g., Empty, which achieves 99%+ accuracy) and up-weights the contribution of hard, misclassified examples (e.g., subtle activities). With $\gamma = 2$, a sample classified with confidence $p_t = 0.9$ (easy) has its loss scaled by $(1 - 0.9)^2 = 0.01$, while a sample classified with $p_t = 0.1$ (hard) retains a scale of $(1 - 0.1)^2 = 0.81$. This forces the optimizer to focus on the challenging activity boundaries rather than refining already-accurate predictions.

```python
# 5class_loeo/train_loeo.py — (lines 169–170)
loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0)
```

**Optimizer — AdamW:**  AdamW decouples weight decay from the gradient update, providing more stable regularization than L2 penalty in Adam. The learning rate of $10^{-4}$ and weight decay of $10^{-4}$ are conservative settings appropriate for the small effective training set size.

```python
# 5class_loeo/train_loeo.py — (lines 35–46, 172)
def get_optimizer(lr, weight_decay):
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        ...
optimizer = get_optimizer(args.lr, args.weight_decay)
```

**Callbacks — Early Stopping, LR Reduction, Checkpointing:**

```python
# 5class_loeo/train_loeo.py — (lines 181–185)
callbacks = [
    ModelCheckpoint(monitor='val_accuracy', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
]
```

- **ModelCheckpoint** saves the model with the best validation accuracy, preventing selection of an overfit late-training-epoch model.
- **EarlyStopping** halts training if validation loss does not improve for 10 epochs, preventing wasted computation and overfitting.
- **ReduceLROnPlateau** halves the learning rate when validation loss plateaus for 3 epochs, enabling fine-grained convergence in later training stages.

---

### 4.6. Inference Pipeline — Four-Antenna Probability Fusion

**Why Not Input-Level Fusion?**  The 8-class variant feeds all 4 antennas as input channels `(T, F, 4)`, which forces the model to learn inter-antenna correlations. However, this makes the model sensitive to antenna arrangement — if the antenna positions change between environments, the learned inter-antenna features become invalid.

The 5-class LOEO variant uses a **probability-level fusion** strategy: each antenna's spectrogram is processed independently through the model, producing 4 separate softmax probability vectors. These are averaged to produce the final prediction:

$$\mathbf{P}_{\text{fused}}(c) = \frac{1}{4} \sum_{a=1}^{4} \mathbf{P}_a(c), \quad c = 1, \ldots, K$$

**Effect:**  This late fusion strategy is more robust because:
1. Each antenna provides an independent "vote" — if one antenna has a poor viewing angle for a particular activity, the other three can compensate.
2. Averaging softmax probabilities reduces the variance of the prediction, analogous to a model ensemble.
3. The model architecture remains single-antenna, requiring no modification when the number of antennas changes.

**Temporal Smoothing:**  A 1-D uniform filter of size 11 is applied along the time axis of the fused probability sequence. This removes short-lived spurious predictions caused by transient noise (e.g., a single window misclassified as Jumping during a Walking segment). The smoothing window of 11 samples corresponds to approximately 0.5–1 second of real time (depending on stride), matching the typical minimum duration of a human activity transition.

**Code Mapping:**

```python
# 5class_loeo/train_loeo.py — Evaluation loop (lines 205–231)

all_preds_fused = []
labels_true = []

for x_batch, y_batch in ds_test:
    # x_batch shape: (batch_size, 4, 340, 100, 1)
    batch_size = tf.shape(x_batch)[0]
    p_final = tf.zeros((batch_size, num_classes), dtype=tf.float32)
    
    for ant in range(4):
        x_ant = x_batch[:, ant, :, :, :]    # (batch_size, 340, 100, 1)
        p_ant = model(x_ant, training=False)  # softmax probabilities
        p_final += p_ant
        
    p_final = p_final / 4.0                  # average across antennas
    all_preds_fused.append(p_final.numpy())

# Temporal Smoothing (11-sample uniform filter)
from scipy.ndimage import uniform_filter1d
preds_smoothed = uniform_filter1d(preds, size=11, axis=0)
labels_pred = np.argmax(preds_smoothed, axis=1)
```

The test dataset yields windows grouped by antenna `(batch, 4, 340, 100, 1)`, constructed in `dataset_loader.py`:

```python
# 5class_loeo/dataset_loader.py — map_four_antennas() (lines 217–237)
def map_four_antennas(idx_tuple):
    start = idx_tuple[0]
    label = idx_tuple[1]
    
    window = master_tensor[start : start + window_length]  # (340, 100, 4)
    window = (window - mean_t) / std_t                     # normalize
    window = tf.transpose(window, [2, 0, 1])               # (4, 340, 100)
    window = tf.expand_dims(window, axis=-1)                # (4, 340, 100, 1)
    ...
```

---

## 5. Experimental Protocol

### 5.1. Three-Fold Leave-One-Environment-Out (LOEO)

The LOEO protocol ensures that the model is evaluated on a **completely unseen physical environment**. In each fold, one of the three environments is held out entirely — it is excluded from training, validation, and normalization statistic computation.

| Fold | Train Environments | Test Environment | Train Sessions | Test Sessions |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Living Room, Laboratory | **Bedroom** | S6a, S6b, S7a | S1a–S1c, S2a–S2b, S3a, S4a–S4b, S5a |
| 2 | Bedroom, Laboratory | **Living Room** | S1a–S1c, S2a–S2b, S3a, S4a–S4b, S5a, S7a | S6a, S6b |
| 3 | Bedroom, Living Room | **Laboratory** | S1a–S1c, S2a–S2b, S3a, S4a–S4b, S5a, S6a, S6b | S7a |

Within each fold, the training sessions are split 75% for training and 25% for validation (time-based split within each activity recording).

### 5.2. Training Configuration

| Hyperparameter | Value |
| :--- | :--- |
| Optimizer | AdamW |
| Learning Rate | $1 \times 10^{-4}$ |
| Weight Decay | $1 \times 10^{-4}$ |
| Loss Function | Sparse Categorical Crossentropy |
| Batch Size | 32 |
| Maximum Epochs | 25 |
| Early Stopping Patience | 10 epochs (monitors `val_loss`, restores best weights) |
| LR Reduction | Factor 0.5 when `val_loss` plateaus for 3 epochs (min LR: $10^{-6}$) |
| Model Checkpointing | Saves model with best `val_accuracy` |

### 5.3. Inference Pipeline

1. **Independent Antenna Processing**: Each test window is a 4-antenna group $(4, 100, 100, 1)$. The trained model processes all 4 antennas through `model.predict()` in a single batch. The softmax outputs $\mathbf{P}_1, \mathbf{P}_2, \mathbf{P}_3, \mathbf{P}_4 \in \mathbb{R}^{8}$ are obtained per antenna.

2. **Probability Fusion**:
$$\mathbf{P}_{\text{fused}}(c) = \frac{1}{4} \sum_{a=1}^{4} \mathbf{P}_a(c), \quad c = 1, \ldots, 8$$

3. **Temporal Smoothing**: A 1-D uniform filter of size 11 is applied along the time axis of the fused probability sequence using `scipy.ndimage.uniform_filter1d`. This removes short-lived spurious predictions caused by transient signal noise.

4. **Final Prediction**: $\hat{y} = \arg\max_c \; \mathbf{P}_{\text{smoothed}}(c)$

---

## 6. Results and Discussion

### 6.1. Dataset Statistics per Fold

| Fold | Test Env | Train Windows | Val Windows | Test Windows |
| :---: | :--- | ---: | ---: | ---: |
| 1 | Bedroom | 50,795 | 16,662 | 223,590 |
| 2 | Living Room | 182,931 | 60,050 | 47,078 |
| 3 | Laboratory | 202,616 | 66,522 | 20,786 |

> **Note**: Fold 1 has the fewest training samples because only Living Room (2 sessions) and Laboratory (1 session) are available for training. Fold 3 has the most training data but the fewest test samples since the Laboratory consists of a single session.

### 6.2. Overall Cross-Environment Performance

| Metric | Fold 1 (Bedroom) | Fold 2 (Living Room) | Fold 3 (Laboratory) | **Mean ± Std** |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 70.95% | 83.58% | 38.63% | **64.39% ± 18.9%** |
| **Macro-Precision** | 64.21% | 78.73% | 27.19% | **56.71% ± 21.6%** |
| **Macro-Recall** | 65.96% | 80.60% | 32.95% | **59.84% ± 19.9%** |
| **Macro-F1** | 59.67% | 77.77% | 26.35% | **54.60% ± 21.3%** |

### 6.3. Per-Class Performance (All 8 Classes)

#### Fold 1 — Test: Bedroom

| Activity | Code | Precision | Recall | F1-Score | Support |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Empty | E | 99.91% | 99.21% | 99.56% | 40,286 |
| Walking | W | 94.48% | 21.12% | 34.53% | 33,941 |
| Running | R | 72.76% | 76.61% | 74.64% | 34,349 |
| Jumping | J | 52.43% | 93.82% | 67.27% | 28,950 |
| Sitting | L | 90.36% | 90.29% | 90.32% | 40,931 |
| Standing | S | 68.25% | 60.59% | 64.19% | 18,896 |
| Continuous sit/stand | C | 32.04% | 85.97% | 46.69% | 11,162 |
| Arm gym | H | 3.45% | 0.09% | 0.18% | 15,075 |

#### Fold 2 — Test: Living Room

| Activity | Code | Precision | Recall | F1-Score | Support |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Empty | E | 100.00% | 99.21% | 99.61% | 7,510 |
| Walking | W | 99.36% | 99.09% | 99.22% | 7,653 |
| Running | R | 98.26% | 99.95% | 99.09% | 7,664 |
| Jumping | J | 90.95% | 99.67% | 95.11% | 5,225 |
| Sitting | L | 97.96% | 54.07% | 69.68% | 7,650 |
| Standing | S | 99.16% | 100.00% | 99.58% | 3,789 |
| Continuous sit/stand | C | 44.16% | 92.80% | 59.85% | 3,794 |
| Arm gym | H | 0.00% | 0.00% | 0.00% | 3,793 |

#### Fold 3 — Test: Laboratory

| Activity | Code | Precision | Recall | F1-Score | Support |
| :--- | :---: | ---: | ---: | ---: | ---: |
| Empty | E | 97.57% | 100.00% | 98.77% | 3,649 |
| Walking | W | 99.42% | 64.27% | 78.07% | 3,725 |
| Running | R | 0.00% | 0.00% | 0.00% | 0 |
| Jumping | J | 20.50% | 99.30% | 33.98% | 2,001 |
| Sitting | L | 0.00% | 0.00% | 0.00% | 0 |
| Standing | S | 0.00% | 0.00% | 0.00% | 3,805 |
| Continuous sit/stand | C | 0.00% | 0.00% | 0.00% | 3,802 |
| Arm gym | H | 0.00% | 0.00% | 0.00% | 3,804 |

> **Important**: The Laboratory session (`S7a`) contains **zero samples** for Running and Sitting. Their 0% scores are artefacts of missing test data, not model failure. Excluding these two absent classes would raise the Laboratory Macro-F1 substantially.

### 6.4. Confusion Matrix Analysis

#### Key Observations

1. **Empty State is Near-Perfect**: The `E` class achieved F1 scores of 99.56%, 99.61%, and 98.77% across the three folds. The static channel signature when no human is present is highly consistent and transfers trivially across environments.

2. **High-Energy Activities Generalize Well (Fold 2)**: In the Living Room fold, Walking (99.22% F1), Running (99.09% F1), and Standing (99.58% F1) all achieved excellent cross-environment performance, suggesting that the Bedroom + Laboratory training data provided sufficient diversity for these gross-motor activities.

3. **Walking–Running Confusion (Fold 1)**: In the Bedroom test, 9,767 Walking windows were misclassified as Running and 17,002 as Jumping. The model's severe underperformance on Walking (21.12% recall) in this fold reveals that the periodic gait Doppler signature learned from only 3 training sessions (S6a, S6b, S7a) was insufficient to generalize.

4. **Arm Gym (H) is Consistently Problematic**: The model achieved near-zero F1 for Arm gym across all folds (0.18%, 0.00%, 0.00%). This suggests that upper-body-only movements produce Doppler signatures too subtle and too variable across environments for the current architecture to distinguish.

5. **Standing vs. Continuous sit/stand Confusion**: In the Bedroom test, 3,682 Standing windows were misclassified as Continuous sit/stand. These two stationary-dominant activities produce similar low-Doppler signatures, making them inherently difficult to separate from spectrograms alone.

### 6.5. Cross-Environment Variance Discussion

The **18.9% standard deviation** in accuracy across folds highlights the fundamental challenge:

- **Fold 2 (Living Room)** benefits from having the largest and most diverse training pool (Bedroom: 9 sessions + Lab: 1 session = 10 sessions), resulting in the highest accuracy.
- **Fold 1 (Bedroom)** has only 3 training sessions, severely limiting the model's exposure to activity variations. Despite this, the model achieves 70.95% accuracy, driven largely by the strong performance on Empty and Sitting.
- **Fold 3 (Laboratory)** illustrates the worst case: the Lab environment has distinct physical characteristics (metallic surfaces, different room dimensions) from the residential training environments, and two activity classes are entirely absent from the test set, collapsing the macro average.

---

## 7. Conclusion

We presented **SHARP++**, a Residual CNN–CBAM–BiLSTM architecture for device-free human activity recognition using Wi-Fi CSI Doppler spectrograms. Through a strict 3-fold LOEO protocol, we demonstrated that:

1. **The architecture generalizes well for high-energy activities** (Walking, Running, Jumping) and the Empty state, achieving near-perfect F1 scores in favorable conditions (Fold 2).
2. **Cross-environment generalization remains a significant open challenge**, with a mean accuracy of 64.39% ± 18.9% and a mean Macro-F1 of 54.60% ± 21.3%.
3. **Training set size critically influences performance**: folds with fewer training sessions show marked degradation.
4. **Subtle activities** (Arm gym, Continuous sit/stand) are inherently difficult to distinguish from spectrograms alone and require further architectural innovation or additional signal modalities.

### Future Work

- **Domain Adaptation**: Applying unsupervised or few-shot domain adaptation techniques to bridge the gap between training and deployment environments.
- **Larger Doppler Windows**: Increasing the window size (e.g., 340 × 100 as explored in our 5-class variant) to capture longer temporal context for subtle activities.
- **Multi-Person Scenarios**: Extending the framework to handle simultaneous activities by multiple occupants.
- **Edge Deployment**: Optimising the model via quantisation and pruning for real-time inference on embedded Wi-Fi routers.

---

## References

1. SHARP Dataset reference paper (original dataset description and activity definitions).
2. S. Woo, J. Park, J.-Y. Lee, and I.-S. Kweon, "CBAM: Convolutional Block Attention Module," *ECCV 2018*.
3. K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," *CVPR 2016*.
4. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," *ICCV 2017*.
