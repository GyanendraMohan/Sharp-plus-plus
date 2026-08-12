"""
SHARP++ 5-Class LOEO — Fold 3 (Kaggle Self-Contained)
======================================================
Test Environment : Laboratory (S7a)
Train Environments: Bedroom + Living Room (S1a-S5a, S6a, S6b)

Upload your doppler_traces/ folder as a Kaggle dataset, then set
DOPPLER_DIR below to match the path.

Enable GPU in Kaggle: Settings → Accelerator → GPU T4 x2 (or P100)
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — Change these for your Kaggle setup
# ═══════════════════════════════════════════════════════════════════════════════
DOPPLER_DIR = '/kaggle/input/sharp-dataset/doppler_traces'  # ← adjust to your dataset path
OUTPUT_DIR  = '/kaggle/working/fold_3'

FOLD_N         = 3
TEST_ENV       = 'Laboratory'
TRAIN_SESSIONS = ['S1a', 'S1b', 'S1c', 'S2a', 'S2b', 'S3a', 'S4a', 'S4b', 'S5a', 'S6a', 'S6b']
TEST_SESSIONS  = ['S7a']

WINDOW_LENGTH = 340
STRIDE        = 17
BATCH_SIZE    = 32
EPOCHS        = 25
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.5
LSTM_UNITS    = 128
DENSE_UNITS   = 256
PATIENCE      = 10
NUM_ANTENNAS  = 4

CLASS_NAMES = ["Empty", "Sitting", "Walking", "Running", "Jumping"]
NUM_CLASSES = len(CLASS_NAMES)

_NAME_TO_PREFIX = {
    "Empty": "E", "Sitting": "L", "Walking": "W", "Running": "R", "Jumping": "J"
}
_ACTIVITY_FILE_VARIANTS = {
    'C': ['C'], 'E': ['E'], 'H': ['H'], 'J': ['J', 'J1', 'J2'],
    'L': ['L'], 'R': ['R'], 'S': ['S'], 'W': ['W'],
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# GPU SETUP (Kaggle NVIDIA)
# ═══════════════════════════════════════════════════════════════════════════════
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[INFO] GPU detected: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("[INFO] Mixed precision (float16) enabled")
else:
    print("[WARN] No GPU detected — running on CPU")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

def residual_block(x, filters, name_prefix='res'):
    shortcut = x
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), padding='same', use_bias=False,
        name=f'{name_prefix}_conv1')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = tf.keras.layers.ReLU(name=f'{name_prefix}_relu1')(x)

    x = tf.keras.layers.Conv2D(
        filters, (3, 3), padding='same', use_bias=False,
        name=f'{name_prefix}_conv2')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, (1, 1), padding='same', use_bias=False,
            name=f'{name_prefix}_proj_conv')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(
            name=f'{name_prefix}_proj_bn')(shortcut)

    x = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, shortcut])
    x = tf.keras.layers.ReLU(name=f'{name_prefix}_relu2')(x)
    return x


def channel_attention(x, reduction_ratio=8, name_prefix='ca'):
    channels = x.shape[-1]
    reduced = max(1, channels // reduction_ratio)

    dense_down = tf.keras.layers.Dense(
        reduced, activation='relu', use_bias=False, name=f'{name_prefix}_dense_down')
    dense_up = tf.keras.layers.Dense(
        channels, use_bias=False, name=f'{name_prefix}_dense_up')

    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name=f'{name_prefix}_avg_pool')(x)
    avg_pool = tf.keras.layers.Reshape((1, 1, channels), name=f'{name_prefix}_avg_reshape')(avg_pool)
    avg_out = dense_up(dense_down(avg_pool))

    max_pool = tf.keras.layers.GlobalMaxPooling2D(name=f'{name_prefix}_max_pool')(x)
    max_pool = tf.keras.layers.Reshape((1, 1, channels), name=f'{name_prefix}_max_reshape')(max_pool)
    max_out = dense_up(dense_down(max_pool))

    scale = tf.keras.layers.Activation('sigmoid', name=f'{name_prefix}_sigmoid')(avg_out + max_out)
    return tf.keras.layers.Multiply(name=f'{name_prefix}_mul')([x, scale])


def spatial_attention(x, kernel_size=7, name_prefix='sa'):
    avg_pool = tf.keras.layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
        name=f'{name_prefix}_avg_pool')(x)
    max_pool = tf.keras.layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
        name=f'{name_prefix}_max_pool')(x)
    concat = tf.keras.layers.Concatenate(axis=-1,
                                         name=f'{name_prefix}_concat')([avg_pool, max_pool])
    attn_map = tf.keras.layers.Conv2D(
        1, kernel_size, padding='same', use_bias=False,
        name=f'{name_prefix}_conv')(concat)
    attn_map = tf.keras.layers.Activation('sigmoid', name=f'{name_prefix}_sigmoid')(attn_map)
    return tf.keras.layers.Multiply(name=f'{name_prefix}_mul')([x, attn_map])


def cbam_block(x, reduction_ratio=8, spatial_kernel=7, name_prefix='cbam'):
    x = channel_attention(x, reduction_ratio, name_prefix=f'{name_prefix}_ch')
    x = spatial_attention(x, spatial_kernel, name_prefix=f'{name_prefix}_sp')
    return x


def build_sharp_plus_plus(input_shape, num_classes,
                           lstm_units=128, dense_units=256, dropout_rate=0.5):
    inputs = tf.keras.Input(shape=input_shape, name='spectrogram_input')

    # Stem
    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', use_bias=False, name='stem_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.ReLU(name='stem_relu')(x)

    # Stage 1
    x = residual_block(x, 64, name_prefix='res1')
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(x)

    # Stage 2
    x = residual_block(x, 128, name_prefix='res2')
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)

    # Stage 3
    x = residual_block(x, 256, name_prefix='res3')

    # CBAM
    x = cbam_block(x, reduction_ratio=8, spatial_kernel=7)

    # Reshape for LSTM
    freq_channels = x.shape[2] * x.shape[3]
    x = tf.keras.layers.Reshape((-1, freq_channels), name='reshape_lstm')(x)

    # Bi-LSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True,
                             dropout=0.2, recurrent_dropout=0.0),
        name='bilstm_1')(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True,
                             dropout=0.2, recurrent_dropout=0.0),
        name='bilstm_2')(x)

    # GAP + Dense + Output
    x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu', name='dense_embed')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout')(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax', dtype='float32', name='output')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='SHARP_plus_plus')


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw_stream(filepath):
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)
    data = data - np.mean(data, axis=0, keepdims=True)
    return data.T.astype(np.float32)


def load_four_antennas(session_dir, session_name, activity_suffix, num_antennas=4):
    streams = []
    for ant in range(num_antennas):
        fname = f"{session_name}_{activity_suffix}_stream_{ant}.txt"
        path  = os.path.join(session_dir, fname)
        if not os.path.exists(path):
            return None
        streams.append(load_raw_stream(path))
    return np.stack(streams, axis=0)


def prepare_dataset(doppler_dir, session_list, classes, window_length, stride,
                    train_ratio=0.75, val_ratio=0.25, num_antennas=4):
    label_map = {act: idx for idx, act in enumerate(classes)}
    giant_list, train_indices, val_indices, test_indices = [], [], [], []
    current_offset = 0

    for session in session_list:
        session_dir = os.path.join(doppler_dir, session)
        if not os.path.isdir(session_dir):
            continue
        for act_name in classes:
            prefix = _NAME_TO_PREFIX[act_name]
            variants = _ACTIVITY_FILE_VARIANTS.get(prefix, [prefix])
            label_idx = label_map[act_name]

            for variant in variants:
                data_4ant = load_four_antennas(session_dir, session, variant, num_antennas)
                if data_4ant is None:
                    continue
                data = np.transpose(data_4ant, (2, 1, 0)).astype(np.float16)
                time_len = data.shape[0]
                giant_list.append(data)

                t_end = int(time_len * train_ratio)
                v_end = int(time_len * (train_ratio + val_ratio))

                for start in range(0, t_end - window_length + 1, stride):
                    for ant in range(num_antennas):
                        train_indices.append((current_offset + start, label_idx, ant))
                for start in range(t_end, v_end - window_length + 1, stride):
                    for ant in range(num_antennas):
                        val_indices.append((current_offset + start, label_idx, ant))
                for start in range(v_end, time_len - window_length + 1, stride):
                    test_indices.append((current_offset + start, label_idx))
                current_offset += time_len

    if not giant_list:
        raise RuntimeError(f"No data loaded from {doppler_dir}")

    giant_array = np.concatenate(giant_list, axis=0)
    input_shape = (window_length, giant_array.shape[1], 1)
    train_idx = np.array(train_indices, dtype=np.int32)
    val_idx   = np.array(val_indices, dtype=np.int32)
    test_idx  = np.array(test_indices, dtype=np.int32)

    print(f"\n[INFO] Dataset built: Master={giant_array.shape} ({giant_array.nbytes/1e9:.2f}GB)")
    print(f"       Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")
    return giant_array, train_idx, val_idx, test_idx, input_shape


def compute_norm_stats(giant_array, indices, window_length, max_samples=5000):
    if len(indices) == 0:
        return (np.zeros((1, giant_array.shape[1], 1), dtype=np.float32),
                np.ones((1, giant_array.shape[1], 1), dtype=np.float32))
    n = min(max_samples, len(indices))
    sample_idxs = indices[np.random.choice(len(indices), n, replace=False)]
    samples = []
    for (start, _, ant) in sample_idxs:
        samples.append(giant_array[start:start+window_length, :, ant:ant+1])
    samples = np.stack(samples, axis=0).astype(np.float64)
    mean = samples.mean(axis=(0, 1), keepdims=True)[0].astype(np.float32)
    std  = np.maximum(samples.std(axis=(0, 1), keepdims=True), 1e-8)[0].astype(np.float32)
    print(f"[INFO] Norm stats computed from {n} sampled windows.")
    return mean, std


def build_normalized_tf_dataset(giant_array, indices, window_length, batch_size, mean, std,
                                mode='train', shuffle=False, repeat=True, augment=False,
                                num_classes=5):
    master_tensor = tf.constant(giant_array)
    mean_t = tf.constant(mean, dtype=tf.float32)
    std_t  = tf.constant(std, dtype=tf.float32)

    if len(indices) == 0:
        cols = 3 if mode in ['train', 'val'] else 2
        indices = np.empty((0, cols), dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle:
        ds = ds.shuffle(min(len(indices), 20000), reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()

    if mode in ['train', 'val']:
        def map_single_antenna(idx_tuple):
            start, label, ant = idx_tuple[0], idx_tuple[1], idx_tuple[2]
            window = master_tensor[start:start+window_length, :, ant:ant+1]
            window = tf.cast(window, tf.float32)
            window = (window - mean_t) / std_t
            return window, tf.one_hot(label, depth=num_classes)
        ds = ds.map(map_single_antenna, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            def augment_fn(x, label):
                x = x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.05)
                def mask_time():
                    win_len = tf.shape(x)[0]
                    mask_size = tf.random.uniform([], 1, 30, dtype=tf.int32)
                    mask_start = tf.random.uniform([], 0, win_len - mask_size, dtype=tf.int32)
                    idx = tf.range(win_len)
                    mask = tf.logical_or(idx < mask_start, idx >= mask_start + mask_size)
                    return x * tf.reshape(tf.cast(mask, tf.float32), [win_len, 1, 1])
                x = tf.cond(tf.random.uniform([]) < 0.3, mask_time, lambda: x)
                return x, label
            ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        def map_four_antennas(idx_tuple):
            start, label = idx_tuple[0], idx_tuple[1]
            window = master_tensor[start:start+window_length]
            window = tf.cast(window, tf.float32)
            window = (window - mean_t) / std_t
            window = tf.transpose(window, [2, 0, 1])
            window = tf.expand_dims(window, axis=-1)
            return window, tf.one_hot(label, depth=num_classes)
        ds = ds.map(map_four_antennas, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

def get_optimizer(lr, weight_decay):
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        pass
    try:
        return tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        pass
    print("  [WARN] AdamW not found — using Adam.")
    return tf.keras.optimizers.Adam(learning_rate=lr)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — FOLD 3
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'#'*70}")
    print(f"  FOLD {FOLD_N}  |  Test Environment: {TEST_ENV}")
    print(f"  Train sessions: {TRAIN_SESSIONS}")
    print(f"  Test  sessions: {TEST_SESSIONS}")
    print(f"{'#'*70}")

    # Filter available sessions
    train_sessions = [s for s in TRAIN_SESSIONS if os.path.isdir(os.path.join(DOPPLER_DIR, s))]
    test_sessions  = [s for s in TEST_SESSIONS  if os.path.isdir(os.path.join(DOPPLER_DIR, s))]

    missing_train = set(TRAIN_SESSIONS) - set(train_sessions)
    missing_test  = set(TEST_SESSIONS) - set(test_sessions)
    if missing_train:
        print(f"  [WARN] Missing train sessions: {sorted(missing_train)}")
    if missing_test:
        print(f"  [WARN] Missing test sessions: {sorted(missing_test)}")

    if not train_sessions or not test_sessions:
        print("[ERROR] Cannot run — missing sessions!")
        return

    # ── Load training data ────────────────────────────────────────────────────
    print(f"\n[Fold {FOLD_N}] Loading training sessions ...")
    giant_tr, train_idx, val_idx, _, input_shape = prepare_dataset(
        DOPPLER_DIR, train_sessions, CLASS_NAMES, WINDOW_LENGTH, STRIDE,
        train_ratio=0.75, val_ratio=0.25,
    )

    # ── Load test data ────────────────────────────────────────────────────────
    print(f"\n[Fold {FOLD_N}] Loading test sessions ...")
    giant_te, _, _, test_idx, _ = prepare_dataset(
        DOPPLER_DIR, test_sessions, CLASS_NAMES, WINDOW_LENGTH, STRIDE,
        train_ratio=0.0, val_ratio=0.0,
    )

    # ── Normalization (training data ONLY) ────────────────────────────────────
    print(f"\n[Fold {FOLD_N}] Computing normalization stats ...")
    mean, std = compute_norm_stats(giant_tr, train_idx, WINDOW_LENGTH, max_samples=5000)
    with open(os.path.join(OUTPUT_DIR, 'norm_stats.pkl'), 'wb') as fp:
        pickle.dump({'mean': mean, 'std': std}, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # ── Build tf.data pipelines ───────────────────────────────────────────────
    print(f"\n[Fold {FOLD_N}] Building tf.data pipelines ...")
    ds_train = build_normalized_tf_dataset(
        giant_tr, train_idx, WINDOW_LENGTH, BATCH_SIZE, mean, std,
        mode='train', shuffle=True, repeat=True, augment=True, num_classes=NUM_CLASSES)
    ds_val = build_normalized_tf_dataset(
        giant_tr, val_idx, WINDOW_LENGTH, BATCH_SIZE, mean, std,
        mode='val', shuffle=False, repeat=True, num_classes=NUM_CLASSES)
    ds_test = build_normalized_tf_dataset(
        giant_te, test_idx, WINDOW_LENGTH, BATCH_SIZE, mean, std,
        mode='test', shuffle=False, repeat=False, num_classes=NUM_CLASSES)

    steps_tr = max(1, int(np.ceil(len(train_idx) / BATCH_SIZE)))
    steps_va = max(1, int(np.ceil(len(val_idx)   / BATCH_SIZE)))

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"\n[Fold {FOLD_N}] Building SHARP++ model ...")
    tf.keras.backend.clear_session()
    model = build_sharp_plus_plus(
        input_shape=input_shape, num_classes=NUM_CLASSES,
        lstm_units=LSTM_UNITS, dense_units=DENSE_UNITS, dropout_rate=DROPOUT)

    loss_fn   = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0)
    optimizer = get_optimizer(LR, WEIGHT_DECAY)
    model.compile(
        optimizer=optimizer, loss=loss_fn,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
    model.summary(line_length=90)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    model_path = os.path.join(OUTPUT_DIR, 'model.keras')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=max(3, PATIENCE // 3),
            min_lr=1e-6, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[Fold {FOLD_N}] Training (up to {EPOCHS} epochs) ...")
    history = model.fit(
        ds_train, epochs=EPOCHS, steps_per_epoch=steps_tr,
        validation_data=ds_val, validation_steps=steps_va,
        callbacks=callbacks, verbose=1)

    with open(os.path.join(OUTPUT_DIR, 'history.json'), 'w') as fp:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, fp, indent=2)

    # ── Evaluate with 4-antenna fusion ────────────────────────────────────────
    print(f"\n[Fold {FOLD_N}] Evaluating on {TEST_ENV} with Probability Fusion ...")
    all_preds_fused, labels_true = [], []

    for x_batch, y_batch in ds_test:
        batch_size = tf.shape(x_batch)[0]
        p_final = tf.zeros((batch_size, NUM_CLASSES), dtype=tf.float32)
        for ant in range(4):
            x_ant = x_batch[:, ant, :, :, :]
            p_ant = model(x_ant, training=False)
            p_final += p_ant
        p_final = p_final / 4.0
        all_preds_fused.append(p_final.numpy())
        labels_true.append(np.argmax(y_batch.numpy(), axis=-1))

    preds = np.concatenate(all_preds_fused, axis=0)
    labels_true = np.concatenate(labels_true, axis=0)

    # Temporal smoothing
    from scipy.ndimage import uniform_filter1d
    preds_smoothed = uniform_filter1d(preds, size=11, axis=0)
    labels_pred = np.argmax(preds_smoothed, axis=1)

    # Metrics
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
    idx = np.arange(NUM_CLASSES)
    conf_matrix = confusion_matrix(labels_true, labels_pred, labels=idx)
    prec, rec, f1, supp = precision_recall_fscore_support(
        labels_true, labels_pred, labels=idx, zero_division=0)
    accuracy = accuracy_score(labels_true, labels_pred)

    print(f"\n{'='*70}")
    print(f"  Fold {FOLD_N} — Test: {TEST_ENV}")
    print(f"{'='*70}")
    print(f"  Overall Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Macro Precision   : {np.mean(prec):.4f}")
    print(f"  Macro Recall      : {np.mean(rec):.4f}")
    print(f"  Macro F1-Score    : {np.mean(f1):.4f}")
    print(f"{'─'*70}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:<20} P={prec[i]:.4f}  R={rec[i]:.4f}  F1={f1[i]:.4f}  N={supp[i]}")
    print(f"{'─'*70}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: {CLASS_NAMES}")
    print(conf_matrix)
    print(f"{'='*70}")

    # Save results
    results = {
        'fold': FOLD_N, 'test_env': TEST_ENV,
        'accuracy': float(accuracy),
        'macro_prec': float(np.mean(prec)), 'macro_rec': float(np.mean(rec)),
        'macro_f1': float(np.mean(f1)),
        'precision': prec.tolist(), 'recall': rec.tolist(), 'fscore': f1.tolist(),
        'conf_matrix': conf_matrix.tolist(),
        'classes': CLASS_NAMES,
        'n_train': len(train_idx), 'n_val': len(val_idx), 'n_test': len(test_idx),
    }
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
