"""
SHARP++ Dataset Loader
======================
Reads raw Doppler stream files (S*_activity_stream_*.txt) directly from
SHARP_plus_plus/doppler_traces/<session>/ subfolders, generates sliding-window
samples, and builds tf.data pipelines for training and evaluation.

Default data location:
    SHARP_plus_plus/doppler_traces/<session>/
        <session>_<activity>_stream_0.txt   ← pickled (time_steps, freq_bins)
        <session>_<activity>_stream_1.txt
        <session>_<activity>_stream_2.txt
        <session>_<activity>_stream_3.txt

Generated window cache:
    SHARP_plus_plus/windows_cache/<session>_<activity>_<idx>.pkl
    (Created once; reused on subsequent runs.)

Pipeline
--------
    Raw stream file  (time_steps × freq_bins)
        → mean-subtract (remove static background)
        → transpose     (freq_bins × time_steps)
        → stack 4 antennas → (4 × freq_bins × total_time)
        → sliding window   → (4 × freq_bins × window_size)
        → transpose        → (window_size × freq_bins × 4)   ← model input
        → 60 / 20 / 20 temporal split per activity
        → tf.data.Dataset
"""

import os
import glob
import pickle
import numpy as np
import tensorflow as tf
from collections import defaultdict


try:
    from dotenv import load_dotenv
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_THIS_DIR, '.env'))
except ImportError:
    pass

_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOPPLER_DIR = os.getenv('DOPPLER_TRACES_DIR', os.path.join(_THIS_DIR, 'doppler_traces'))
DEFAULT_CACHE_DIR   = os.getenv('WINDOWS_CACHE_DIR', os.path.join(_THIS_DIR, 'windows_cache'))

# ─────────────────────────────────────────────────────────────────────────────
# Activity definitions  (from the SHARP paper)
# ─────────────────────────────────────────────────────────────────────────────

# Full activity name for each label code (as described in the paper)
ACTIVITY_NAMES = {
    'E':  'Empty room',
    'W':  'Walking',
    'R':  'Running',
    'J':  'Jumping in place',
    'L':  'Sitting',
    'S':  'Standing',
    'C':  'Sit-down / Stand-up',
    'H':  'Arm gym',
}

# Default: all 8 activities from the dataset
DEFAULT_ACTIVITIES = 'E,W,R,J,L,S,C,H'

# ─────────────────────────────────────────────────────────────────────────────
# Activity → filename suffix mapping
# ─────────────────────────────────────────────────────────────────────────────
# SHARP uses single-character activity codes in filenames.
# 'J' activity may appear as 'J', 'J1', or 'J2' depending on the session.
# All variants are merged under the same integer label.

_ACTIVITY_FILE_VARIANTS = {
    'C':  ['C'],
    'E':  ['E'],
    'H':  ['H'],
    'J':  ['J', 'J1', 'J2'],   # both jump variants mapped to 'J'
    'L':  ['L'],
    'R':  ['R'],
    'S':  ['S'],
    'W':  ['W'],
}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level: raw stream file loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_stream(filepath):
    """
    Load one raw Doppler stream file.

    Each file is a pickled numpy array with shape (time_steps, freq_bins).

    Returns:
        np.ndarray  float32  (freq_bins, time_steps)  — mean-subtracted & transposed
    """
    with open(filepath, 'rb') as fp:
        data = pickle.load(fp)              # (time_steps, freq_bins)

    # Remove static background (mean over time axis)
    data = data - np.mean(data, axis=0, keepdims=True)

    return data.T.astype(np.float32)       # (freq_bins, time_steps)


def load_four_antennas(session_dir, session_name, activity_suffix, num_antennas=4):
    """
    Load all antenna streams for one activity and stack them.

    Args:
        session_dir:      path to e.g. doppler_traces/S1a/
        session_name:     e.g. 'S1a'
        activity_suffix:  e.g. 'E', 'J1', 'J2'
        num_antennas:     number of streams (default 4)

    Returns:
        np.ndarray  float32  (num_antennas, freq_bins, time_steps)
        or None if any stream file is missing
    """
    streams = []
    for ant in range(num_antennas):
        fname = f"{session_name}_{activity_suffix}_stream_{ant}.txt"
        path  = os.path.join(session_dir, fname)
        if not os.path.exists(path):
            return None                    # skip missing activities
        streams.append(load_raw_stream(path))

    return np.stack(streams, axis=0)       # (4, freq_bins, time_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────────────────────────────────────────

def sliding_windows(data_4ant, window_length, stride):
    """
    Apply a sliding window along the time axis.

    Args:
        data_4ant:     np.ndarray  (4, freq_bins, total_time)
        window_length: int  — number of time steps per window
        stride:        int  — step size between windows

    Returns:
        list of np.ndarray  (window_length, freq_bins, 4)  float32
    """
    total_time = data_4ant.shape[2]
    windows    = []
    for start in range(0, total_time - window_length + 1, stride):
        w = data_4ant[:, :, start:start + window_length]   # (4, freq, win)
        w = np.transpose(w, (2, 1, 0))                     # (win, freq, 4)
        windows.append(w.astype(np.float32))
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation: stream files → windowed cache
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(
        doppler_dir   = None,
        session_list  = None,
        activity_str  = DEFAULT_ACTIVITIES,
        window_length = 100,
        stride        = 5,
        train_ratio   = 0.60,
        val_ratio     = 0.20,
        cache_dir     = None,
        num_antennas  = 4,
        force_rebuild = False,
):
    """
    Read raw Doppler stream files, generate sliding-window samples, cache them
    to disk, and return train / val / test file-path + label lists.

    Args:
        doppler_dir:   path to doppler_traces/ folder
                       (default: SHARP_plus_plus/doppler_traces/)
        session_list:  list of session names, e.g. ['S1a', 'S1b']
                       (default: all subdirectories found in doppler_dir)
        activity_str:  comma-separated activity labels, e.g. 'E,L,W,R,J'
        window_length: time steps per window          (default: 100)
        stride:        stride between windows          (default: 5)
        train_ratio:   fraction for training           (default: 0.60)
        val_ratio:     fraction for validation         (default: 0.20)
        cache_dir:     where to save windowed .pkl files
                       (default: SHARP_plus_plus/windows_cache/)
        num_antennas:  number of antenna streams       (default: 4)
        force_rebuild: if True, delete existing cache and rebuild

    Returns:
        train_files, train_labels,
        val_files,   val_labels,
        test_files,  test_labels,
        input_shape  — tuple (window_length, freq_bins, num_antennas)
    """
    # ── Defaults ──────────────────────────────────────────────────────────────
    if doppler_dir is None:
        doppler_dir = DEFAULT_DOPPLER_DIR
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    doppler_dir = os.path.abspath(doppler_dir)
    cache_dir   = os.path.abspath(cache_dir)

    # Discover sessions if not specified
    if session_list is None:
        session_list = sorted([
            d for d in os.listdir(doppler_dir)
            if os.path.isdir(os.path.join(doppler_dir, d))
        ])
    if not session_list:
        raise RuntimeError(f"[ERROR] No session directories found in: {doppler_dir}")

    activities = [a.strip() for a in activity_str.split(',')]
    label_map  = {act: idx for idx, act in enumerate(activities)}

    os.makedirs(cache_dir, exist_ok=True)

    # ── Build cache tag for this configuration ─────────────────────────────
    # A unique subdirectory per (sessions + activities + window + stride) combo
    tag = (f"{'_'.join(sorted(session_list))}"
           f"_act{activity_str.replace(',','')}"
           f"_win{window_length}_str{stride}")
    split_cache = os.path.join(cache_dir, tag)

    # ── Generate windowed files if not already cached ──────────────────────
    already_cached = (os.path.isdir(split_cache) and
                      len(glob.glob(os.path.join(split_cache, '*.pkl'))) > 0)

    if already_cached and not force_rebuild:
        print(f"[INFO] Using cached windows from: {split_cache}")
    else:
        if force_rebuild and os.path.isdir(split_cache):
            import shutil
            shutil.rmtree(split_cache)
        os.makedirs(split_cache, exist_ok=True)
        print(f"[INFO] Generating windowed samples → {split_cache}")
        _generate_windows(
            doppler_dir, session_list, activities, label_map,
            window_length, stride, split_cache, num_antennas
        )

    # ── Load the split index ───────────────────────────────────────────────
    return _build_splits(
        split_cache, activities, label_map,
        train_ratio, val_ratio
    )


def _generate_windows(doppler_dir, session_list, activities, label_map,
                       window_length, stride, split_cache, num_antennas):
    """Read raw stream files and write windowed .pkl samples to split_cache."""
    total_written = 0

    for session in session_list:
        session_dir = os.path.join(doppler_dir, session)
        if not os.path.isdir(session_dir):
            print(f"[WARN] Session directory not found, skipping: {session_dir}")
            continue

        for act_label in activities:
            # Each label can map to multiple file-name variants (e.g. J→J1,J2)
            variants = _ACTIVITY_FILE_VARIANTS.get(act_label, [act_label])
            label_idx = label_map[act_label]

            for variant in variants:
                data_4ant = load_four_antennas(session_dir, session, variant, num_antennas)
                if data_4ant is None:
                    continue     # this variant doesn't exist for this session

                windows = sliding_windows(data_4ant, window_length, stride)
                if not windows:
                    print(f"[WARN] No windows generated for {session}/{act_label}/{variant}")
                    continue

                for idx, w in enumerate(windows):
                    fname = f"{session}_{act_label}_{variant}_{idx:05d}.pkl"
                    fpath = os.path.join(split_cache, fname)
                    with open(fpath, 'wb') as fp:
                        pickle.dump({'data': w, 'label': label_idx}, fp,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                    total_written += 1

                print(f"    {session}  {act_label}/{variant:<3}  "
                      f"{len(windows):>5} windows  "
                      f"shape={windows[0].shape}")

    print(f"[INFO] Total windows written: {total_written}")
    if total_written == 0:
        raise RuntimeError(
            f"[ERROR] No windows were generated!\n"
            f"        doppler_dir = {doppler_dir}\n"
            f"        sessions    = {session_list}\n"
            f"        activities  = {activities}\n"
            f"        Expected files like: "
            f"{session_list[0]}/{session_list[0]}_E_stream_0.txt"
        )


def _build_splits(split_cache, activities, label_map, train_ratio, val_ratio):
    """
    Load all cached .pkl paths, group by (session, activity, variant),
    apply a temporal 60/20/20 split per group, and return flat lists.

    The temporal split preserves time-order within each group, preventing
    train/test leakage that would occur with random shuffling.
    """
    # Group files by (session, activity) — extract from filename
    # Filename pattern: {session}_{act_label}_{variant}_{idx:05d}.pkl
    groups = defaultdict(list)
    for fname in sorted(os.listdir(split_cache)):
        if not fname.endswith('.pkl'):
            continue
        parts  = fname.rsplit('_', 1)          # split off index
        prefix = parts[0]                       # e.g. S1a_E_E
        groups[prefix].append(os.path.join(split_cache, fname))

    # Sort each group by index (already sorted by filename, but be explicit)
    for key in groups:
        groups[key].sort()

    train_files, train_labels = [], []
    val_files,   val_labels   = [], []
    test_files,  test_labels  = [], []

    for prefix, paths in sorted(groups.items()):
        # Extract label from first file
        with open(paths[0], 'rb') as fp:
            lbl = pickle.load(fp)['label']

        n     = len(paths)
        t_end = int(n * train_ratio)
        v_end = int(n * (train_ratio + val_ratio))

        train_files  += paths[:t_end];  train_labels  += [lbl] * t_end
        val_files    += paths[t_end:v_end]; val_labels += [lbl] * (v_end - t_end)
        test_files   += paths[v_end:];  test_labels   += [lbl] * (n - v_end)

    # Detect input_shape from first available sample
    # (train_files may be empty when train_ratio=0.0, e.g. LOEO test-env mode)
    first_file = (train_files or val_files or test_files)[0]
    with open(first_file, 'rb') as fp:
        sample_data = pickle.load(fp)['data']
    input_shape = sample_data.shape   # (window_length, freq_bins, num_antennas)

    print(f"\n[INFO] Dataset split summary:")
    print(f"       Input shape : {input_shape}  (time × freq × antennas)")
    print(f"       Train       : {len(train_files)} samples")
    print(f"       Val         : {len(val_files)} samples")
    print(f"       Test        : {len(test_files)} samples")

    return (train_files, train_labels,
            val_files,   val_labels,
            test_files,  test_labels,
            input_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level sample loader (used by tf.data)
# ─────────────────────────────────────────────────────────────────────────────

def load_window_file(filepath):
    """Load one cached window .pkl file and return (data, label)."""
    if isinstance(filepath, (bytes, bytearray)):
        filepath = filepath.decode('utf-8')
    with open(filepath, 'rb') as fp:
        rec = pickle.load(fp)
    return rec['data'].astype(np.float32)


def load_multi_antenna(filepath):
    """
    Alias kept for compatibility — same as load_window_file (returns data only).
    """
    return load_window_file(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# tf.data pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_tf_dataset(file_list, labels, input_shape, batch_size,
                     shuffle, cache_file=None, prefetch=True, repeat=True):
    """
    Build a tf.data.Dataset from cached window .pkl file paths.

    Args:
        file_list:   list[str]  — paths to .pkl window files
        labels:      list[int]  — integer class labels
        input_shape: tuple      — (window_length, freq_bins, num_antennas)
        batch_size:  int
        shuffle:     bool       — enable shuffle (True for training only)
        cache_file:  str|None   — tf.data cache path prefix (None = no cache)
        prefetch:    bool
        repeat:      bool       — repeat indefinitely (True for train/val)

    Returns:
        tf.data.Dataset  yielding (spectrogram_batch, label_batch)
    """
    dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))

    def _load_fn(filepath, label):
        data = tf.ensure_shape(
            tf.numpy_function(load_window_file, [filepath], tf.float32),
            input_shape
        )
        return data, label

    dataset = dataset.map(_load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_file is not None:
        dataset = dataset.cache(cache_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(labels), 5000),
                                  reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers  (used by LOEO protocol)
# ─────────────────────────────────────────────────────────────────────────────

def compute_norm_stats(file_list, max_samples=5000):
    """
    Compute per-element mean and standard deviation from a sample of training
    window files using Welford's online algorithm (numerically stable).

    Stats have the same shape as one input sample: (window, freq, antennas).
    They should be computed ONLY from training data to prevent data leakage.

    Args:
        file_list:   list[str]  — training window .pkl file paths
        max_samples: int        — cap on files sampled for efficiency

    Returns:
        mean : np.ndarray  float32  (window, freq, antennas)
        std  : np.ndarray  float32  (window, freq, antennas)  — clamped ≥ 1e-8
    """
    import random

    n_sample = min(max_samples, len(file_list))
    sampled  = random.sample(file_list, n_sample)

    n    = 0
    mean = None
    M2   = None

    for path in sampled:
        with open(path, 'rb') as fp:
            x = pickle.load(fp)['data'].astype(np.float64)
        n += 1
        if mean is None:
            mean = np.zeros(x.shape, dtype=np.float64)
            M2   = np.zeros(x.shape, dtype=np.float64)
        delta  = x - mean
        mean  += delta / n
        delta2 = x - mean
        M2    += delta * delta2

    variance = M2 / max(n - 1, 1)
    std      = np.sqrt(variance)
    std      = np.maximum(std, 1e-8)     # prevent division by zero

    print(f"[INFO] Norm stats (n={n}): "
          f"mean={float(mean.mean()):.5f}  std={float(std.mean()):.5f}")

    return mean.astype(np.float32), std.astype(np.float32)


def build_normalized_tf_dataset(file_list, labels, input_shape, batch_size,
                                 mean, std,
                                 shuffle=False, repeat=True):
    """
    Build a tf.data.Dataset that loads .pkl window files and applies
    z-score normalization inline: x_norm = (x - mean) / std.

    Args:
        file_list:   list[str]  — window .pkl file paths
        labels:      list[int]  — integer class labels
        input_shape: tuple      — (window_length, freq_bins, num_antennas)
        batch_size:  int
        mean:        np.ndarray (input_shape) — training set mean
        std:         np.ndarray (input_shape) — training set std
        shuffle:     bool       — True for training split only
        repeat:      bool       — True for train/val; False for test

    Returns:
        tf.data.Dataset  yielding (normalized_batch, label_batch)
    """
    mean_t = tf.constant(mean, dtype=tf.float32)
    std_t  = tf.constant(std,  dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))

    def _load_and_norm(filepath, label):
        data = tf.ensure_shape(
            tf.numpy_function(load_window_file, [filepath], tf.float32),
            input_shape
        )
        data = (data - mean_t) / std_t
        return data, label

    dataset = dataset.map(_load_and_norm, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(labels), 5000),
                                  reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    doppler = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOPPLER_DIR
    session = sys.argv[2] if len(sys.argv) > 2 else None
    acts    = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_ACTIVITIES
    sessions = [session] if session else None

    print(f"[TEST] doppler_dir  = {doppler}")
    print(f"[TEST] sessions     = {sessions or 'auto-discover'}")
    print(f"[TEST] activities   = {acts}")
    print()

    result = prepare_dataset(
        doppler_dir  = doppler,
        session_list = sessions,
        activity_str = acts,
        window_length= 100,
        stride       = 5,
    )

    tr_f, tr_l, va_f, va_l, te_f, te_l, shape = result

    # Load one sample to verify
    sample = load_window_file(tr_f[0])
    print(f"\n[TEST] Sample shape : {sample.shape},  dtype: {sample.dtype}")
    print(f"[TEST] Value range  : [{sample.min():.4f}, {sample.max():.4f}]")
    print(f"[TEST] Label sample : {tr_l[0]}")
    print(f"\n[OK]  dataset_loader.py sanity check passed.")
