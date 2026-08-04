"""
SHARP++ Training Script
=======================
Trains the Residual CNN + CBAM + Bi-LSTM model on WiFi Doppler spectrograms.
Reads raw stream files (S*_activity_stream_*.txt) from doppler_traces/.

Usage
-----
    python train.py [OPTIONS]

All arguments are optional — sensible defaults point to SHARP_plus_plus/doppler_traces/

Optional arguments
    --doppler_dir  STR   Path to doppler_traces/ folder
                         (default: ./doppler_traces/)
    --sessions     STR   Comma-separated session dirs, e.g. S1a,S1b,S1c
                         (default: all sessions found in doppler_dir)
    --activities   STR   Comma-separated activity labels  (default: E,W,R,J,L,S,C,H)
    --window       INT   Time steps per spectrogram window (default: 100)
    --stride       INT   Stride between windows            (default: 5)
    --batch_size   INT   Batch size                        (default: 32)
    --epochs       INT   Max training epochs               (default: 100)
    --lr           FLOAT Learning rate                     (default: 1e-4)
    --weight_decay FLOAT AdamW weight decay                (default: 1e-4)
    --dropout      FLOAT Dropout rate                      (default: 0.5)
    --lstm_units   INT   LSTM hidden units / direction     (default: 128)
    --dense_units  INT   Dense embedding size              (default: 256)
    --patience     INT   Early-stopping patience           (default: 10)
    --output_dir   STR   Where to save model & results     (default: ./outputs/)
    --rebuild      FLAG  Force rebuild of window cache

Example
-------
    cd SHARP_plus_plus

    # Train on all sessions found in ./doppler_traces/
    python train.py

    # Train on specific sessions and activities
    python train.py --sessions S1a,S1b,S1c --activities E,L,W,R,J

    # Train with custom hyperparameters
    python train.py --sessions S1a --epochs 50 --batch_size 16
"""

import argparse
import json
import os
import sys
import numpy as np
import pickle
import tensorflow as tf

# Ensure imports resolve from SHARP_plus_plus/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import build_sharp_plus_plus
from dataset_loader import prepare_dataset, build_tf_dataset, ACTIVITY_NAMES, DEFAULT_ACTIVITIES

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_optimizer(lr, weight_decay):
    """Return AdamW if available, fall back to Adam."""
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        pass
    try:
        return tf.keras.optimizers.experimental.AdamW(
            learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        pass
    print("[WARN] AdamW not found — using Adam.")
    return tf.keras.optimizers.Adam(learning_rate=lr)


def _steps(n, batch_size):
    return int(np.ceil(n / batch_size))


def _clean_cache(prefix):
    """Remove leftover tf.data cache files."""
    for ext in ['.data-00000-of-00001', '.index']:
        p = prefix + ext
        if os.path.exists(p):
            os.remove(p)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train SHARP++ (Residual CNN + CBAM + Bi-LSTM)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--doppler_dir',  default=os.path.join(_THIS_DIR, 'doppler_traces'),
                        help='Path to doppler_traces/ folder')
    parser.add_argument('--sessions',     default=None,
                        help='Comma-separated session dirs (default: all in doppler_dir)')
    parser.add_argument('--activities',   default=DEFAULT_ACTIVITIES,
                        help='Comma-separated activity labels (default: all 8)')
    parser.add_argument('--window',       type=int,   default=100,
                        help='Time steps per spectrogram window')
    parser.add_argument('--stride',       type=int,   default=5,
                        help='Stride between windows')
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout',      type=float, default=0.5)
    parser.add_argument('--lstm_units',   type=int,   default=128)
    parser.add_argument('--dense_units',  type=int,   default=256)
    parser.add_argument('--patience',     type=int,   default=10)
    parser.add_argument('--output_dir',   default=os.path.join(_THIS_DIR, 'outputs'))
    parser.add_argument('--rebuild',      action='store_true',
                        help='Force rebuild of the window cache')
    args = parser.parse_args()

    # ── GPU / Metal setup (Apple Silicon M1/M2/M3/M4) ─────────────────────────
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[INFO] Metal GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("[INFO] Mixed precision (float16) enabled for Metal GPU")
    else:
        print("[INFO] GPUs available: [] — running on CPU.")

    # ── Config ────────────────────────────────────────────────────────────────
    session_list  = args.sessions.split(',') if args.sessions else None
    activity_str  = args.activities
    activities    = activity_str.split(',')
    num_classes   = len(activities)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SHARP++ Training")
    print(f"{'='*60}")
    print(f"  Doppler dir : {args.doppler_dir}")
    print(f"  Sessions    : {session_list or 'auto-discover'}")
    print(f"  Activities  : {activities}  ({num_classes} classes)")
    print(f"  Window      : {args.window} steps   Stride: {args.stride}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Max epochs  : {args.epochs}")
    print(f"  LR / WD     : {args.lr} / {args.weight_decay}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Prepare windowed dataset ───────────────────────────────────────────────
    print("[INFO] Preparing windowed dataset from raw stream files ...")
    (train_files, train_labels,
     val_files,   val_labels,
     test_files,  test_labels,
     input_shape) = prepare_dataset(
        doppler_dir   = args.doppler_dir,
        session_list  = session_list,
        activity_str  = activity_str,
        window_length = args.window,
        stride        = args.stride,
        force_rebuild = args.rebuild,
    )

    print(f"\n[INFO] Input shape : {input_shape}  (time × freq × antennas)")

    # ── Build tf.data pipelines ───────────────────────────────────────────────
    name_base  = os.path.join(args.output_dir, f'sharp_pp_{activity_str}')

    for split in ['train', 'val', 'test']:
        _clean_cache(f"{name_base}_cache_{split}")

    dataset_train = build_tf_dataset(
        train_files, train_labels, input_shape, args.batch_size,
        shuffle=True, cache_file=f"{name_base}_cache_train", repeat=True
    )
    dataset_val = build_tf_dataset(
        val_files, val_labels, input_shape, args.batch_size,
        shuffle=False, cache_file=f"{name_base}_cache_val", repeat=True
    )
    dataset_test = build_tf_dataset(
        test_files, test_labels, input_shape, args.batch_size,
        shuffle=False, cache_file=None, repeat=False
    )

    steps_train = _steps(len(train_files), args.batch_size)
    steps_val   = _steps(len(val_files),   args.batch_size)

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n[INFO] Building SHARP++ model ...")
    model = build_sharp_plus_plus(
        input_shape  = input_shape,
        num_classes  = num_classes,
        lstm_units   = args.lstm_units,
        dense_units  = args.dense_units,
        dropout_rate = args.dropout,
    )
    model.summary(line_length=90)

    # ── Compile ───────────────────────────────────────────────────────────────
    optimizer = get_optimizer(args.lr, args.weight_decay)
    model.compile(
        optimizer = optimizer,
        loss      = tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics   = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    model_path = f"{name_base}_network.keras"
    callbacks = [
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

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[INFO] Training for up to {args.epochs} epochs "
          f"(early stopping patience={args.patience}) ...")
    history = model.fit(
        dataset_train,
        epochs           = args.epochs,
        steps_per_epoch  = steps_train,
        validation_data  = dataset_val,
        validation_steps = steps_val,
        callbacks        = callbacks,
    )

    # ── Save history ──────────────────────────────────────────────────────────
    history_path = f"{name_base}_history.json"
    with open(history_path, 'w') as fp:
        json.dump({k: [float(v) for v in vs]
                   for k, vs in history.history.items()}, fp, indent=2)
    print(f"[INFO] History → {history_path}")
    print(f"[INFO] Model   → {model_path}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n[INFO] Evaluating on test set ...")
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

    labels_true  = np.array(test_labels)
    predictions  = model.predict(dataset_test)[:labels_true.shape[0]]
    labels_pred  = np.argmax(predictions, axis=1)

    idx          = np.arange(num_classes)
    conf_matrix  = confusion_matrix(labels_true, labels_pred, labels=idx)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_true, labels_pred, labels=idx, zero_division=0)
    accuracy = accuracy_score(labels_true, labels_pred)

    print(f"\n{'='*60}")
    print(f"  SHARP++ — Final Test Results")
    print(f"{'='*60}")
    print(f"  Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Macro F1 : {np.mean(f1):.4f}")
    print(f"{'─'*65}")
    print(f"  {'Code':<6} {'Activity':<22} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"{'─'*65}")
    for i, act in enumerate(activities):
        name = ACTIVITY_NAMES.get(act, act)
        print(f"  {act:<6} {name:<22} {prec[i]:>8.4f} {rec[i]:>8.4f} {f1[i]:>8.4f}")
    print(f"{'─'*65}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: {activities}")
    print(conf_matrix)
    print(f"{'='*60}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'conf_matrix': conf_matrix,
        'accuracy': float(accuracy),
        'precision': prec, 'recall': rec, 'fscore': f1,
        'activities': activities, 'input_shape': input_shape,
    }
    with open(f"{name_base}_test_results.pkl", 'wb') as fp:
        pickle.dump(results, fp)
    print(f"[INFO] Results → {name_base}_test_results.pkl")


if __name__ == '__main__':
    main()
