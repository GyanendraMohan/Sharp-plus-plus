"""
SHARP++ — Leave-One-Environment-Out (LOEO) Protocol
=====================================================
Trains and evaluates the Residual CNN + CBAM + Bi-LSTM model using a
three-fold LOEO cross-validation over the three SHARP environments:
    Fold 1  →  Test: Bedroom      (S1–S5),  Train: Living Room + Laboratory
    Fold 2  →  Test: Living Room  (S6),     Train: Bedroom + Laboratory
    Fold 3  →  Test: Laboratory   (S7),     Train: Bedroom + Living Room

Environment → Measurement Set Mapping (SHARP paper, Table 1):
    Bedroom    : S1a–S1c, S2a–S2b, S3a, S4a–S4b, S5a
    Living Room: S6a, S6b
    Laboratory : S7a

Key protocol rules enforced here:
    1. No data from the test environment appears in training or validation.
    2. Normalization statistics computed ONLY from training data per fold.
    3. Identical preprocessing pipeline across all folds.
    4. Per-fold results + aggregate mean ± std reported.

Usage
-----
    cd /Users/gyanendramohanpatel/Desktop/SHARP_plus_plus
    source .venv/bin/activate

    # Run all 3 folds (default)
    python SHARP_plus_plus/loeo.py

    # Run specific fold(s)
    python SHARP_plus_plus/loeo.py --folds 2,3

    # Custom paths / hyperparameters
    python SHARP_plus_plus/loeo.py \\
        --doppler_dir SHARP_plus_plus/doppler_traces/ \\
        --activities E,W,R,J,L,S,C,H \\
        --epochs 100 \\
        --batch_size 32 \\
        --lstm_units 128 \\
        --dense_units 256

Output files (in --output_dir, default: SHARP_plus_plus/outputs/loeo/):
    fold_<N>/model.keras          — best model for fold N
    fold_<N>/history.json         — training curves
    fold_<N>/results.pkl          — pickled per-fold metrics dict
    fold_<N>/norm_stats.pkl       — normalization mean/std (to reproduce)
    aggregate_results.json        — mean ± std across all folds
"""

import argparse
import json
import os
import pickle
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_sharp_plus_plus
from dataset_loader import (
    prepare_dataset,
    compute_norm_stats,
    build_normalized_tf_dataset,
    ACTIVITY_NAMES,
    DEFAULT_ACTIVITIES,
    DEFAULT_DOPPLER_DIR,
    DEFAULT_CACHE_DIR,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Fold definitions  (matches SHARP paper Table 1)
# ─────────────────────────────────────────────────────────────────────────────

LOEO_FOLDS = [
    {
        'fold': 1,
        'test_env':       'Bedroom',
        'train_envs':     ['Living Room', 'Laboratory'],
        'train_sessions': ['S6a', 'S6b', 'S7a'],
        'test_sessions':  ['S1a', 'S1b', 'S1c',
                           'S2a', 'S2b',
                           'S3a',
                           'S4a', 'S4b',
                           'S5a'],
    },
    {
        'fold': 2,
        'test_env':       'Living Room',
        'train_envs':     ['Bedroom', 'Laboratory'],
        'train_sessions': ['S1a', 'S1b', 'S1c',
                           'S2a', 'S2b',
                           'S3a',
                           'S4a', 'S4b',
                           'S5a',
                           'S7a'],
        'test_sessions':  ['S6a', 'S6b'],
    },
    {
        'fold': 3,
        'test_env':       'Laboratory',
        'train_envs':     ['Bedroom', 'Living Room'],
        'train_sessions': ['S1a', 'S1b', 'S1c',
                           'S2a', 'S2b',
                           'S3a',
                           'S4a', 'S4b',
                           'S5a',
                           'S6a', 'S6b'],
        'test_sessions':  ['S7a'],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def filter_available(sessions, doppler_dir):
    """Return only sessions whose directory exists in doppler_dir."""
    available = [
        s for s in sessions
        if os.path.isdir(os.path.join(doppler_dir, s))
    ]
    missing = set(sessions) - set(available)
    if missing:
        print(f"  [WARN] Sessions not found (skipped): {sorted(missing)}")
    return available


def get_optimizer(lr, weight_decay):
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        pass
    try:
        return tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        pass
    print("  [WARN] AdamW not found — using Adam.")
    return tf.keras.optimizers.Adam(learning_rate=lr)


def steps(n, batch_size):
    return max(1, int(np.ceil(n / batch_size)))


def _clean_tf_cache(prefix):
    for ext in ['.data-00000-of-00001', '.index']:
        p = prefix + ext
        if os.path.exists(p):
            os.remove(p)


# ─────────────────────────────────────────────────────────────────────────────
# Single-fold runner
# ─────────────────────────────────────────────────────────────────────────────

def run_fold(fold_cfg, args):
    """
    Execute one LOEO fold:
      1. Prepare windowed data for training and test environments.
      2. Compute normalization stats from training data only.
      3. Train the model with the normalized pipeline.
      4. Evaluate on the completely held-out test environment.
      5. Save model, history, normalization stats, and metrics.

    Args:
        fold_cfg:  dict from LOEO_FOLDS
        args:      argparse.Namespace

    Returns:
        dict with per-fold evaluation metrics
    """
    fold_n    = fold_cfg['fold']
    test_env  = fold_cfg['test_env']
    fold_dir  = os.path.join(args.output_dir, f'fold_{fold_n}')
    os.makedirs(fold_dir, exist_ok=True)

    activities  = args.activities.split(',')
    num_classes = len(activities)

    print(f"\n{'#'*70}")
    print(f"  FOLD {fold_n}  |  Test Environment: {test_env}")
    print(f"  Train envs : {fold_cfg['train_envs']}")
    print(f"{'#'*70}")

    # ── Filter sessions that actually exist on disk ────────────────────────
    train_sessions = filter_available(fold_cfg['train_sessions'], args.doppler_dir)
    test_sessions  = filter_available(fold_cfg['test_sessions'],  args.doppler_dir)

    if not train_sessions:
        print(f"  [ERROR] No training sessions found for fold {fold_n}. Skipping.")
        return None
    if not test_sessions:
        print(f"  [ERROR] No test sessions found for fold {fold_n}. Skipping.")
        return None

    print(f"  Train sessions ({len(train_sessions)}): {train_sessions}")
    print(f"  Test  sessions ({len(test_sessions)}): {test_sessions}")

    # ── Prepare training data (75% train / 25% val, no test portion) ──────
    print(f"\n[Fold {fold_n}] Preparing training environment windows ...")
    (tr_files, tr_labels,
     va_files, va_labels,
     _te_f,   _te_l,
     input_shape) = prepare_dataset(
        doppler_dir   = args.doppler_dir,
        session_list  = train_sessions,
        activity_str  = args.activities,
        window_length = args.window,
        stride        = args.stride,
        train_ratio   = 0.75,
        val_ratio     = 0.25,
        cache_dir     = args.cache_dir,
        force_rebuild = args.rebuild,
    )
    print(f"  Train: {len(tr_files)}  Val: {len(va_files)}")

    # ── Prepare test data (ALL windows from test environment) ──────────────
    print(f"\n[Fold {fold_n}] Preparing test environment windows ...")
    (_tr_f, _tr_l,
     _va_f, _va_l,
     te_files, te_labels,
     _shape) = prepare_dataset(
        doppler_dir   = args.doppler_dir,
        session_list  = test_sessions,
        activity_str  = args.activities,
        window_length = args.window,
        stride        = args.stride,
        train_ratio   = 0.0,          # no training split from test env
        val_ratio     = 0.0,          # no validation split from test env
        cache_dir     = args.cache_dir,
        force_rebuild = args.rebuild,
    )
    print(f"  Test : {len(te_files)}")

    if not te_files:
        print(f"  [ERROR] Test dataset empty for fold {fold_n}. Skipping.")
        return None

    # ── Normalization (from training data ONLY) ────────────────────────────
    print(f"\n[Fold {fold_n}] Computing normalization stats from training data ...")
    mean, std = compute_norm_stats(tr_files, max_samples=5000)

    # Save norm stats for reproducibility
    norm_path = os.path.join(fold_dir, 'norm_stats.pkl')
    with open(norm_path, 'wb') as fp:
        pickle.dump({'mean': mean, 'std': std}, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Norm stats → {norm_path}")

    # ── Build tf.data pipelines (normalized) ──────────────────────────────
    tr_cache = os.path.join(fold_dir, 'cache_train')
    va_cache = os.path.join(fold_dir, 'cache_val')
    for p in [tr_cache, va_cache]:
        _clean_tf_cache(p)

    ds_train = build_normalized_tf_dataset(
        tr_files, tr_labels, input_shape, args.batch_size, mean, std,
        shuffle=True, repeat=True
    )
    ds_val = build_normalized_tf_dataset(
        va_files, va_labels, input_shape, args.batch_size, mean, std,
        shuffle=False, repeat=True
    )
    ds_test = build_normalized_tf_dataset(
        te_files, te_labels, input_shape, args.batch_size, mean, std,
        shuffle=False, repeat=False
    )

    steps_tr = steps(len(tr_files), args.batch_size)
    steps_va = steps(len(va_files), args.batch_size)

    # ── Build model ────────────────────────────────────────────────────────
    print(f"\n[Fold {fold_n}] Building SHARP++ model ...")
    tf.keras.backend.clear_session()   # fresh graph per fold

    model = build_sharp_plus_plus(
        input_shape  = input_shape,
        num_classes  = num_classes,
        lstm_units   = args.lstm_units,
        dense_units  = args.dense_units,
        dropout_rate = args.dropout,
    )

    optimizer = get_optimizer(args.lr, args.weight_decay)
    model.compile(
        optimizer = optimizer,
        loss      = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics   = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    )
    model.summary(line_length=85)

    # ── Callbacks ──────────────────────────────────────────────────────────
    model_path = os.path.join(fold_dir, 'model.keras')
    callbacks  = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=args.patience,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=max(3, args.patience // 3),
            min_lr=1e-6, verbose=1
        ),
    ]

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n[Fold {fold_n}] Training (up to {args.epochs} epochs, "
          f"patience={args.patience}) ...")
    history = model.fit(
        ds_train,
        epochs           = args.epochs,
        steps_per_epoch  = steps_tr,
        validation_data  = ds_val,
        validation_steps = steps_va,
        callbacks        = callbacks,
        verbose          = 1,
    )

    # Save history
    history_path = os.path.join(fold_dir, 'history.json')
    with open(history_path, 'w') as fp:
        json.dump({k: [float(v) for v in vs]
                   for k, vs in history.history.items()}, fp, indent=2)
    print(f"  History → {history_path}")

    # ── Evaluate on test environment ────────────────────────────────────────
    print(f"\n[Fold {fold_n}] Evaluating on held-out {test_env} ...")
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
        accuracy_score,
    )

    labels_true = np.array(te_labels)
    preds       = model.predict(ds_test, verbose=1)[:labels_true.shape[0]]
    labels_pred = np.argmax(preds, axis=1)

    idx          = np.arange(num_classes)
    conf_matrix  = confusion_matrix(labels_true, labels_pred, labels=idx)
    prec, rec, f1, supp = precision_recall_fscore_support(
        labels_true, labels_pred, labels=idx, zero_division=0)
    accuracy = accuracy_score(labels_true, labels_pred)

    # ── Print per-fold results ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Fold {fold_n} — Test: {test_env}")
    print(f"{'='*70}")
    print(f"  Overall Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Macro Precision   : {np.mean(prec):.4f}")
    print(f"  Macro Recall      : {np.mean(rec):.4f}")
    print(f"  Macro F1-Score    : {np.mean(f1):.4f}")
    print(f"{'─'*70}")
    print(f"  {'Code':<5} {'Activity':<22} {'Prec':>8} {'Rec':>8} {'F1':>8} {'N':>7}")
    print(f"{'─'*70}")
    for i, act in enumerate(activities):
        name = ACTIVITY_NAMES.get(act, act)
        print(f"  {act:<5} {name:<22} {prec[i]:>8.4f} {rec[i]:>8.4f} "
              f"{f1[i]:>8.4f} {supp[i]:>7}")
    print(f"{'─'*70}")
    print(f"\n  Confusion Matrix  (rows=true, cols=pred)")
    print(f"  Labels: {activities}")
    print(conf_matrix)
    print(f"{'='*70}")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        'fold':         fold_n,
        'test_env':     test_env,
        'train_envs':   fold_cfg['train_envs'],
        'accuracy':     float(accuracy),
        'precision':    prec,
        'recall':       rec,
        'fscore':       f1,
        'macro_prec':   float(np.mean(prec)),
        'macro_rec':    float(np.mean(rec)),
        'macro_f1':     float(np.mean(f1)),
        'conf_matrix':  conf_matrix,
        'activities':   activities,
        'input_shape':  input_shape,
        'n_train':      len(tr_files),
        'n_val':        len(va_files),
        'n_test':       len(te_files),
    }
    results_path = os.path.join(fold_dir, 'results.pkl')
    with open(results_path, 'wb') as fp:
        pickle.dump(results, fp)
    print(f"  Results   → {results_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate results across folds
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_and_print(fold_results, output_dir, activities):
    """Compute mean ± std across all completed folds and save to JSON."""
    completed = [r for r in fold_results if r is not None]
    if not completed:
        print("\n[ERROR] No folds completed — no aggregate results.")
        return

    accs  = [r['accuracy']   for r in completed]
    precs = [r['macro_prec'] for r in completed]
    recs  = [r['macro_rec']  for r in completed]
    f1s   = [r['macro_f1']   for r in completed]

    print(f"\n{'#'*70}")
    print(f"  SHARP++ LOEO — Aggregate Results  ({len(completed)} folds)")
    print(f"{'#'*70}")
    print(f"  {'Metric':<18} {'Mean':>10} {'± Std':>10}")
    print(f"  {'─'*40}")
    print(f"  {'Accuracy':<18} {np.mean(accs)*100:>9.2f}%  ± {np.std(accs)*100:.2f}%")
    print(f"  {'Macro Precision':<18} {np.mean(precs):>10.4f}  ± {np.std(precs):.4f}")
    print(f"  {'Macro Recall':<18} {np.mean(recs):>10.4f}  ± {np.std(recs):.4f}")
    print(f"  {'Macro F1-Score':<18} {np.mean(f1s):>10.4f}  ± {np.std(f1s):.4f}")
    print(f"  {'─'*40}")
    print(f"\n  Per-fold breakdown:")
    print(f"  {'Fold':<8} {'Test Env':<16} {'Acc':>8} {'F1':>8} {'N-test':>8}")
    print(f"  {'─'*52}")
    for r in completed:
        print(f"  Fold {r['fold']:<4} {r['test_env']:<16} "
              f"{r['accuracy']*100:>7.2f}% {r['macro_f1']:>8.4f} {r['n_test']:>8}")
    print(f"{'#'*70}")

    # Per-activity average across folds
    print(f"\n  Per-activity Mean F1 across folds:")
    print(f"  {'Code':<5} {'Activity':<22} {'Mean F1':>10}")
    print(f"  {'─'*40}")
    for i, act in enumerate(activities):
        act_f1s = [r['fscore'][i] for r in completed]
        print(f"  {act:<5} {ACTIVITY_NAMES.get(act, act):<22} {np.mean(act_f1s):>10.4f}")
    print(f"  {'─'*40}")

    # Save aggregate JSON
    aggregate = {
        'n_folds':           len(completed),
        'activities':        activities,
        'mean_accuracy':     float(np.mean(accs)),
        'std_accuracy':      float(np.std(accs)),
        'mean_macro_prec':   float(np.mean(precs)),
        'std_macro_prec':    float(np.std(precs)),
        'mean_macro_rec':    float(np.mean(recs)),
        'std_macro_rec':     float(np.std(recs)),
        'mean_macro_f1':     float(np.mean(f1s)),
        'std_macro_f1':      float(np.std(f1s)),
        'per_fold': [
            {
                'fold':      r['fold'],
                'test_env':  r['test_env'],
                'accuracy':  r['accuracy'],
                'macro_f1':  r['macro_f1'],
            }
            for r in completed
        ],
    }
    agg_path = os.path.join(output_dir, 'aggregate_results.json')
    with open(agg_path, 'w') as fp:
        json.dump(aggregate, fp, indent=2)
    print(f"\n  Aggregate results → {agg_path}")

    return aggregate


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SHARP++ LOEO (Leave-One-Environment-Out) Protocol',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--doppler_dir', default=os.path.join(_THIS_DIR, 'doppler_traces'),
                        help='Path to doppler_traces/ folder')
    parser.add_argument('--activities', default=DEFAULT_ACTIVITIES,
                        help='Comma-separated activity labels')
    parser.add_argument('--window',      type=int,   default=100)
    parser.add_argument('--stride',      type=int,   default=5)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--weight_decay',type=float, default=1e-4)
    parser.add_argument('--dropout',     type=float, default=0.5)
    parser.add_argument('--lstm_units',  type=int,   default=128)
    parser.add_argument('--dense_units', type=int,   default=256)
    parser.add_argument('--patience',    type=int,   default=10)
    parser.add_argument('--output_dir',  default=os.path.join(_THIS_DIR, 'outputs', 'loeo'))
    parser.add_argument('--cache_dir',   default=os.path.join(_THIS_DIR, 'windows_cache'))
    parser.add_argument('--folds',       default='1,2,3',
                        help='Comma-separated fold numbers to run, e.g. 1,2,3')
    parser.add_argument('--rebuild',     action='store_true',
                        help='Force rebuild of window cache')
    args = parser.parse_args()

    # ── GPU / Metal setup (Apple Silicon M1/M2/M3/M4) ─────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] Metal GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Mixed float16 precision — ~2× faster on Apple Metal GPU
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("[INFO] Mixed precision (float16) enabled for Metal GPU")
    else:
        print("[WARN] No GPU detected — running on CPU.")
        print("       Install tensorflow-metal:  pip install tensorflow-metal")

    os.makedirs(args.output_dir, exist_ok=True)

    selected_folds = [int(f.strip()) for f in args.folds.split(',')]
    activities     = args.activities.split(',')

    print(f"\n{'='*70}")
    print(f"  SHARP++ LOEO Protocol")
    print(f"{'='*70}")
    print(f"  Doppler dir  : {args.doppler_dir}")
    print(f"  Activities   : {activities}  ({len(activities)} classes)")
    print(f"  Window/Stride: {args.window}/{args.stride}")
    print(f"  Folds        : {selected_folds}")
    print(f"  Batch/Epochs : {args.batch_size}/{args.epochs}")
    print(f"  LR/WD        : {args.lr}/{args.weight_decay}")
    print(f"  LSTM/Dense   : {args.lstm_units}/{args.dense_units}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"{'='*70}")

    # ── Run folds ─────────────────────────────────────────────────────────
    fold_results = []
    for fold_cfg in LOEO_FOLDS:
        if fold_cfg['fold'] not in selected_folds:
            continue
        result = run_fold(fold_cfg, args)
        fold_results.append(result)

    # ── Aggregate ─────────────────────────────────────────────────────────
    aggregate_and_print(fold_results, args.output_dir, activities)


if __name__ == '__main__':
    main()
