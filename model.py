"""
SHARP++ Model
=============
Residual CNN + CBAM Attention + Bi-LSTM for WiFi Human Activity Recognition.

Architecture:
    Input (time × freq × antennas)
    → Stem Conv2D(64) + BN + ReLU
    → ResBlock(64)  + MaxPool(2×2)
    → ResBlock(128) + MaxPool(2×2)
    → ResBlock(256)
    → CBAM (Channel Attention + Spatial Attention)
    → Reshape → (time/4, freq/4 × 256)
    → BiLSTM(128) × 2
    → GlobalAveragePooling1D
    → Dense(256, ReLU)
    → Dropout(0.5)
    → Dense(num_classes, Softmax)
"""

import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# Residual Block
# ─────────────────────────────────────────────────────────────────────────────

def residual_block(x, filters, name_prefix='res'):
    """
    Standard residual block.
      y = F(x) + x   (with 1×1 projection when channel dims differ)

    Args:
        x:           Input tensor
        filters:     Number of output filters
        name_prefix: Prefix for layer names (keep unique per block)

    Returns:
        Output tensor with skip connection added.
    """
    shortcut = x

    # Main path
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


# ─────────────────────────────────────────────────────────────────────────────
# CBAM: Convolutional Block Attention Module
# ─────────────────────────────────────────────────────────────────────────────

def channel_attention(x, reduction_ratio=8, name_prefix='ca'):
    """
    Channel Attention Module.
    Learns which feature channels are important via GlobalAvg + GlobalMax pooling
    and a shared 2-layer MLP.

    Args:
        x:               Input feature map  (H × W × C)
        reduction_ratio: MLP bottleneck ratio
        name_prefix:     Unique layer name prefix

    Returns:
        Channel-recalibrated feature map  (H × W × C)
    """
    channels = x.shape[-1]
    reduced = max(1, channels // reduction_ratio)

    # Shared MLP weights
    dense_down = tf.keras.layers.Dense(
        reduced, activation='relu', use_bias=False, name=f'{name_prefix}_dense_down')
    dense_up = tf.keras.layers.Dense(
        channels, use_bias=False, name=f'{name_prefix}_dense_up')

    # Global Average Pool branch
    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name=f'{name_prefix}_avg_pool')(x)
    avg_pool = tf.keras.layers.Reshape((1, 1, channels), name=f'{name_prefix}_avg_reshape')(avg_pool)
    avg_out = dense_up(dense_down(avg_pool))

    # Global Max Pool branch
    max_pool = tf.keras.layers.GlobalMaxPooling2D(name=f'{name_prefix}_max_pool')(x)
    max_pool = tf.keras.layers.Reshape((1, 1, channels), name=f'{name_prefix}_max_reshape')(max_pool)
    max_out = dense_up(dense_down(max_pool))

    # Combine and gate
    scale = tf.keras.layers.Activation('sigmoid', name=f'{name_prefix}_sigmoid')(avg_out + max_out)
    return tf.keras.layers.Multiply(name=f'{name_prefix}_mul')([x, scale])


def spatial_attention(x, kernel_size=7, name_prefix='sa'):
    """
    Spatial Attention Module.
    Generates a 2-D attention map indicating *where* to focus.

    Args:
        x:           Input feature map  (H × W × C)
        kernel_size: Conv kernel for the attention map (7 recommended)
        name_prefix: Unique layer name prefix

    Returns:
        Spatially-recalibrated feature map  (H × W × C)
    """
    # Channel-wise statistics — must use Lambda to stay inside Keras graph
    avg_pool = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
        name=f'{name_prefix}_avg_pool')(x)                        # H × W × 1
    max_pool = tf.keras.layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
        name=f'{name_prefix}_max_pool')(x)                        # H × W × 1
    concat = tf.keras.layers.Concatenate(axis=-1,
                                         name=f'{name_prefix}_concat')([avg_pool, max_pool])
    # Convolve to spatial map
    attn_map = tf.keras.layers.Conv2D(
        1, kernel_size, padding='same', use_bias=False,
        name=f'{name_prefix}_conv')(concat)
    attn_map = tf.keras.layers.Activation('sigmoid', name=f'{name_prefix}_sigmoid')(attn_map)
    return tf.keras.layers.Multiply(name=f'{name_prefix}_mul')([x, attn_map])


def cbam_block(x, reduction_ratio=8, spatial_kernel=7, name_prefix='cbam'):
    """
    CBAM = Channel Attention → Spatial Attention (applied sequentially).

    Suppresses background noise and static reflections while highlighting
    human Doppler motion signatures.

    Args:
        x:               Input feature map  (H × W × C)
        reduction_ratio: For channel attention MLP
        spatial_kernel:  Kernel size for spatial attention conv
        name_prefix:     Unique layer name prefix

    Returns:
        Attended feature map  (H × W × C)
    """
    x = channel_attention(x, reduction_ratio, name_prefix=f'{name_prefix}_ch')
    x = spatial_attention(x, spatial_kernel, name_prefix=f'{name_prefix}_sp')
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Full SHARP++ Model
# ─────────────────────────────────────────────────────────────────────────────

def build_sharp_plus_plus(input_shape, num_classes,
                           lstm_units=128, dense_units=256, dropout_rate=0.5):
    """
    Build the full SHARP++ model.

    Args:
        input_shape:  (time_steps, freq_bins, num_antennas)  e.g. (340, 100, 4)
        num_classes:  Number of activity classes
        lstm_units:   Hidden units per LSTM direction (total = lstm_units × 2)
        dense_units:  Dense embedding size
        dropout_rate: Dropout probability before output layer

    Returns:
        tf.keras.Model
    """
    inputs = tf.keras.Input(shape=input_shape, name='spectrogram_input')

    # ── Stem ──────────────────────────────────────────────────────────────────
    # input: (T, F, C)  e.g. (340, 100, 4)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', use_bias=False, name='stem_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.ReLU(name='stem_relu')(x)
    # output: (T, F, 64)

    # ── Stage 1: ResBlock(64) + MaxPool ───────────────────────────────────────
    x = residual_block(x, 64, name_prefix='res1')
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(x)
    # output: (T/2, F/2, 64)

    # ── Stage 2: ResBlock(128) + MaxPool ──────────────────────────────────────
    x = residual_block(x, 128, name_prefix='res2')
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
    # output: (T/4, F/4, 128)

    # ── Stage 3: ResBlock(256) ────────────────────────────────────────────────
    x = residual_block(x, 256, name_prefix='res3')
    # output: (T/4, F/4, 256)

    # ── CBAM Attention ────────────────────────────────────────────────────────
    x = cbam_block(x, reduction_ratio=8, spatial_kernel=7)
    # output: (T/4, F/4, 256)  — static reflections suppressed

    # ── Reshape for LSTM ──────────────────────────────────────────────────────
    # Treat height dimension (T/4) as time steps for the LSTM.
    # Each step aggregates spatial-freq features: (F/4 × 256) per step.
    freq_channels = x.shape[2] * x.shape[3]   # F/4 * 256  (static, known at build time)
    x = tf.keras.layers.Reshape(
        (-1, freq_channels), name='reshape_lstm')(x)
    # output: (T/4, F/4 × 256)

    # ── Bi-LSTM Stack ─────────────────────────────────────────────────────────
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True,
                             dropout=0.2, recurrent_dropout=0.0),
        name='bilstm_1'
    )(x)
    # output: (T/4, lstm_units × 2)  e.g. (85, 256)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True,
                             dropout=0.2, recurrent_dropout=0.0),
        name='bilstm_2'
    )(x)
    # output: (T/4, lstm_units × 2)  e.g. (85, 256)

    # ── Global Average Pooling over time ──────────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)
    # output: (lstm_units × 2,)  e.g. (256,)

    # ── Dense Embedding ───────────────────────────────────────────────────────
    x = tf.keras.layers.Dense(
        dense_units, activation='relu', name='dense_embed')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout')(x)
    # output: (dense_units,)  e.g. (256,)

    # ── Classification Head ───────────────────────────────────────────────────
    # dtype='float32' required when mixed_float16 precision is active
    # (softmax must be computed in full precision to avoid numerical issues)
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', dtype='float32', name='output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SHARP_plus_plus')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import numpy as np

    # Actual SHARP dataset dimensions: 100 time steps × 100 freq bins × 4 antennas, 8 classes
    INPUT_SHAPE = (100, 100, 4)
    NUM_CLASSES = 8

    model = build_sharp_plus_plus(INPUT_SHAPE, NUM_CLASSES)
    model.summary(line_length=90)

    # Forward pass smoke test
    dummy = np.random.randn(2, *INPUT_SHAPE).astype(np.float32)
    out = model(dummy, training=False)
    print(f"\n[OK] Output shape: {out.shape}  (expected: (2, {NUM_CLASSES}))")
