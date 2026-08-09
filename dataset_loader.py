"""
SHARP++ Dataset Loader (Dynamic Indexing OOM Fix)
=================================================
Loads raw Doppler stream files into a single, continuous float16 array in memory
to avoid the 90GB+ RAM explosion of duplicating overlapping windows.

tf.data dynamically slices the master tensor on the GPU during training.
"""

import os
import pickle
import numpy as np
import tensorflow as tf

_THIS_DIR           = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOPPLER_DIR = os.path.join(_THIS_DIR, 'doppler_traces')

ACTIVITY_NAMES = {
    'E': 'Empty room', 'W': 'Walking', 'R': 'Running', 'J': 'Jumping in place',
    'L': 'Sitting', 'S': 'Standing', 'C': 'Sit-down / Stand-up', 'H': 'Arm gym',
}

DEFAULT_ACTIVITIES = 'E,W,R,J,L,S,C,H'

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

def prepare_dataset(doppler_dir=None, session_list=None, activity_str=DEFAULT_ACTIVITIES,
                    window_length=100, stride=5, train_ratio=0.75, val_ratio=0.25,
                    num_antennas=4, cache_dir=None, force_rebuild=False):
    """
    Returns:
        giant_array (float16): all time-series concatenated.
        train_indices (int32): (start_idx, label) pairs.
        val_indices   (int32): (start_idx, label) pairs.
        test_indices  (int32): (start_idx, label) pairs.
        input_shape: tuple (window_length, freq_bins, num_antennas)
    """
    doppler_dir = doppler_dir or DEFAULT_DOPPLER_DIR
    session_list = session_list or sorted([d for d in os.listdir(doppler_dir) if os.path.isdir(os.path.join(doppler_dir, d))])
    
    activities = [a.strip() for a in activity_str.split(',')]
    label_map  = {act: idx for idx, act in enumerate(activities)}

    giant_list = []
    train_indices = []
    val_indices = []
    test_indices = []
    
    current_offset = 0

    for session in session_list:
        session_dir = os.path.join(doppler_dir, session)
        if not os.path.isdir(session_dir): continue

        for act_label in activities:
            variants = _ACTIVITY_FILE_VARIANTS.get(act_label, [act_label])
            label_idx = label_map[act_label]

            for variant in variants:
                data_4ant = load_four_antennas(session_dir, session, variant, num_antennas)
                if data_4ant is None: continue

                # Transpose to (time, freq, antennas)
                data = np.transpose(data_4ant, (2, 1, 0)).astype(np.float16)
                time_len = data.shape[0]
                giant_list.append(data)

                t_end = int(time_len * train_ratio)
                v_end = int(time_len * (train_ratio + val_ratio))

                # Train Windows
                for start in range(0, t_end - window_length + 1, stride):
                    train_indices.append((current_offset + start, label_idx))
                # Val Windows
                for start in range(t_end, v_end - window_length + 1, stride):
                    val_indices.append((current_offset + start, label_idx))
                # Test Windows
                for start in range(v_end, time_len - window_length + 1, stride):
                    test_indices.append((current_offset + start, label_idx))

                current_offset += time_len

    if not giant_list:
        raise RuntimeError(f"No data loaded from {doppler_dir}")

    giant_array = np.concatenate(giant_list, axis=0) # e.g. (1.5M, 100, 4) ~ 1.2 GB
    
    input_shape = (window_length, giant_array.shape[1], giant_array.shape[2])
    
    train_idx = np.array(train_indices, dtype=np.int32)
    val_idx   = np.array(val_indices, dtype=np.int32)
    test_idx  = np.array(test_indices, dtype=np.int32)
    
    print(f"\n[INFO] Dataset built in-memory:")
    print(f"       Master Array: {giant_array.shape} ({giant_array.nbytes / 1e9:.2f} GB)")
    print(f"       Train       : {len(train_idx):,} windows")
    print(f"       Val         : {len(val_idx):,} windows")
    print(f"       Test        : {len(test_idx):,} windows")

    return giant_array, train_idx, val_idx, test_idx, input_shape

def compute_norm_stats(giant_array, indices, window_length, max_samples=5000):
    """
    Randomly sample windows using indices to compute a global mean and std.
    Because data is time-shift invariant, we compute a (1, freq_bins, antennas) mean/std
    which broadcasts perfectly.
    """
    if len(indices) == 0:
        return np.zeros((1, giant_array.shape[1], giant_array.shape[2]), dtype=np.float32), \
               np.ones((1, giant_array.shape[1], giant_array.shape[2]), dtype=np.float32)
               
    n = min(max_samples, len(indices))
    sample_idxs = indices[np.random.choice(len(indices), n, replace=False)]
    
    samples = []
    for (start, _) in sample_idxs:
        samples.append(giant_array[start:start+window_length])
        
    samples = np.stack(samples, axis=0).astype(np.float64) # (N, win, freq, 4)
    
    mean = samples.mean(axis=(0, 1), keepdims=True) # (1, 1, freq, 4)
    std  = samples.std(axis=(0, 1), keepdims=True)  # (1, 1, freq, 4)
    std  = np.maximum(std, 1e-8)
    
    # Drop N dimension so it aligns with window dimension (1, freq, 4)
    mean = mean[0].astype(np.float32)
    std  = std[0].astype(np.float32)

    print(f"[INFO] Norm stats computed from {n} sampled windows.")
    return mean, std

def build_normalized_tf_dataset(giant_array, indices, window_length, batch_size, mean, std,
                                shuffle=False, repeat=True, augment=False):
    """
    Builds a tf.data.Dataset using dynamic indexing on the master tensor.
    """
    # 1. Lift variables to TF
    master_tensor = tf.constant(giant_array) # 1.2 GB float16
    mean_t = tf.constant(mean, dtype=tf.float32)
    std_t  = tf.constant(std, dtype=tf.float32)

    # 2. Base dataset is just index pairs (start_idx, label)
    if len(indices) == 0:
        indices = np.empty((0, 2), dtype=np.int32)
    
    ds = tf.data.Dataset.from_tensor_slices(indices)
    
    if shuffle:
        ds = ds.shuffle(min(len(indices), 20000), reshuffle_each_iteration=True)
    
    if repeat:
        ds = ds.repeat()

    # 3. Dynamic slicing + normalization function
    def map_slice_and_norm(idx_pair):
        start = idx_pair[0]
        label = idx_pair[1]
        
        # Slice window directly from master tensor
        window = master_tensor[start : start + window_length]
        window = tf.cast(window, tf.float32)
        
        # Normalize
        window = (window - mean_t) / std_t
        
        return window, label

    # Map the slicing with high parallelization
    ds = ds.map(map_slice_and_norm, num_parallel_calls=tf.data.AUTOTUNE)

    # 4. Augmentation
    if augment:
        def augment_fn(x, label):
            # Noise
            x = x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.05)
            
            # Antenna Dropout
            def drop_antenna():
                ant_idx = tf.random.uniform([], 0, 4, dtype=tf.int32)
                mask = tf.one_hot(ant_idx, 4, on_value=0.0, off_value=1.0)
                return x * tf.reshape(mask, [1, 1, 4])
            x = tf.cond(tf.random.uniform([]) < 0.3, drop_antenna, lambda: x)
            
            # Time Masking
            def mask_time():
                win_len = tf.shape(x)[0]
                mask_size = tf.random.uniform([], 1, 10, dtype=tf.int32)
                mask_start = tf.random.uniform([], 0, win_len - mask_size, dtype=tf.int32)
                idx = tf.range(win_len)
                mask = tf.logical_or(idx < mask_start, idx >= mask_start + mask_size)
                return x * tf.reshape(tf.cast(mask, tf.float32), [win_len, 1, 1])
            x = tf.cond(tf.random.uniform([]) < 0.3, mask_time, lambda: x)
            
            return x, label
        
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
