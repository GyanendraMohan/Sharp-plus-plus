"""
SHARP++ Dataset Loader (5-Class LOEO, 340x100, Independent Antennas)
====================================================================
"""

import os
import pickle
import numpy as np
import tensorflow as tf

_THIS_DIR           = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOPPLER_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'doppler_traces')

# User requested classes
CLASS_NAMES = [
    "Empty",
    "Sitting",
    "Walking",
    "Running",
    "Jumping"
]

# Map names to original SHARP file prefixes
_NAME_TO_PREFIX = {
    "Empty": "E",
    "Sitting": "L",
    "Walking": "W",
    "Running": "R",
    "Jumping": "J"
}

_ACTIVITY_FILE_VARIANTS = {
    'C': ['C'], 'E': ['E'], 'H': ['H'], 'J': ['J', 'J1', 'J2'],
    'L': ['L'], 'R': ['R'], 'S': ['S'], 'W': ['W'],
}

def load_raw_stream(filepath):
    """Returns np.ndarray (freq_bins, time_steps) — mean-subtracted"""
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)
    data = data - np.mean(data, axis=0, keepdims=True)
    return data.T.astype(np.float32)

def load_four_antennas(session_dir, session_name, activity_suffix, num_antennas=4):
    """Returns np.ndarray (4, freq_bins, time_steps) or None"""
    streams = []
    for ant in range(num_antennas):
        fname = f"{session_name}_{activity_suffix}_stream_{ant}.txt"
        path  = os.path.join(session_dir, fname)
        if not os.path.exists(path):
            return None
        streams.append(load_raw_stream(path))
    return np.stack(streams, axis=0)

def prepare_dataset(doppler_dir=None, session_list=None, classes=CLASS_NAMES,
                    window_length=340, stride=17, train_ratio=0.75, val_ratio=0.25,
                    num_antennas=4):
    """
    Returns:
        giant_array (float16): all time-series concatenated (time, freq, 4).
        train_indices (int32): (start_idx, label, ant_idx) — each antenna is an independent sample.
        val_indices   (int32): (start_idx, label, ant_idx) — each antenna is an independent sample.
        test_indices  (int32): (start_idx, label) — keeps all 4 antennas grouped for fusion.
        input_shape: (window_length, freq_bins, 1)
    """
    doppler_dir = doppler_dir or DEFAULT_DOPPLER_DIR
    session_list = session_list or sorted([d for d in os.listdir(doppler_dir) if os.path.isdir(os.path.join(doppler_dir, d))])
    
    label_map  = {act: idx for idx, act in enumerate(classes)}

    giant_list = []
    train_indices = []
    val_indices = []
    test_indices = []
    
    current_offset = 0

    for session in session_list:
        session_dir = os.path.join(doppler_dir, session)
        if not os.path.isdir(session_dir): continue

        for act_name in classes:
            prefix = _NAME_TO_PREFIX[act_name]
            variants = _ACTIVITY_FILE_VARIANTS.get(prefix, [prefix])
            label_idx = label_map[act_name]

            for variant in variants:
                data_4ant = load_four_antennas(session_dir, session, variant, num_antennas)
                if data_4ant is None: continue

                # Transpose to (time, freq, antennas)
                data = np.transpose(data_4ant, (2, 1, 0)).astype(np.float16)
                time_len = data.shape[0]
                giant_list.append(data)

                t_end = int(time_len * train_ratio)
                v_end = int(time_len * (train_ratio + val_ratio))

                # Train Windows (independent antennas)
                for start in range(0, t_end - window_length + 1, stride):
                    for ant in range(num_antennas):
                        train_indices.append((current_offset + start, label_idx, ant))
                # Val Windows (independent antennas)
                for start in range(t_end, v_end - window_length + 1, stride):
                    for ant in range(num_antennas):
                        val_indices.append((current_offset + start, label_idx, ant))
                # Test Windows (grouped for 4-antenna fusion)
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
    
    print(f"\n[INFO] Dataset built in-memory:")
    print(f"       Master Array: {giant_array.shape} ({giant_array.nbytes / 1e9:.2f} GB)")
    print(f"       Train       : {len(train_idx):,} independent single-antenna windows")
    print(f"       Val         : {len(val_idx):,} independent single-antenna windows")
    print(f"       Test        : {len(test_idx):,} 4-antenna grouped windows (for fusion)")

    return giant_array, train_idx, val_idx, test_idx, input_shape

def compute_norm_stats(giant_array, indices, window_length, max_samples=5000):
    """
    Compute mean and std on independent antennas.
    """
    if len(indices) == 0:
        return np.zeros((1, giant_array.shape[1], 1), dtype=np.float32), \
               np.ones((1, giant_array.shape[1], 1), dtype=np.float32)
               
    n = min(max_samples, len(indices))
    sample_idxs = indices[np.random.choice(len(indices), n, replace=False)]
    
    samples = []
    for (start, _, ant) in sample_idxs:
        # Extract specific antenna (time, freq, 1)
        samples.append(giant_array[start:start+window_length, :, ant:ant+1])
        
    samples = np.stack(samples, axis=0).astype(np.float64) # (N, win, freq, 1)
    
    mean = samples.mean(axis=(0, 1), keepdims=True) # (1, 1, freq, 1)
    std  = samples.std(axis=(0, 1), keepdims=True)  # (1, 1, freq, 1)
    std  = np.maximum(std, 1e-8)
    
    # Drop N dimension so it aligns with window dimension (1, freq, 1)
    mean = mean[0].astype(np.float32)
    std  = std[0].astype(np.float32)

    print(f"[INFO] Norm stats computed from {n} sampled windows.")
    return mean, std

def build_normalized_tf_dataset(giant_array, indices, window_length, batch_size, mean, std,
                                mode='train', shuffle=False, repeat=True, augment=False, num_classes=5):
    """
    mode: 'train'/'val' -> yields (window (340, 100, 1), label (one-hot))
    mode: 'test' -> yields (window (4, 340, 100, 1), label (one-hot))
    """
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
            start = idx_tuple[0]
            label = idx_tuple[1]
            ant   = idx_tuple[2]
            
            # Slice window and specific antenna
            window = master_tensor[start : start + window_length, :, ant:ant+1]
            window = tf.cast(window, tf.float32)
            window = (window - mean_t) / std_t
            
            label_one_hot = tf.one_hot(label, depth=num_classes)
            return window, label_one_hot

        ds = ds.map(map_single_antenna, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            def augment_fn(x, label):
                # Noise
                x = x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.05)
                # Time Masking
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
        # Test mode: Group all 4 antennas
        def map_four_antennas(idx_tuple):
            start = idx_tuple[0]
            label = idx_tuple[1]
            
            # (340, 100, 4)
            window = master_tensor[start : start + window_length]
            window = tf.cast(window, tf.float32)
            
            # Normalize each antenna using the same (1, 100, 1) stats 
            # Note: broadcast across the 4 antennas might require slightly careful shape handling
            # window is (340, 100, 4). mean_t is (1, 100, 1). 
            # Broadcasting works automatically!
            window = (window - mean_t) / std_t
            
            # Transpose/reshape to (4, 340, 100, 1) for the model which expects batch of (340, 100, 1)
            # Transpose to (4, 340, 100) -> Reshape to (4, 340, 100, 1)
            window = tf.transpose(window, [2, 0, 1])
            window = tf.expand_dims(window, axis=-1)
            
            label_one_hot = tf.one_hot(label, depth=num_classes)
            return window, label_one_hot

        ds = ds.map(map_four_antennas, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
