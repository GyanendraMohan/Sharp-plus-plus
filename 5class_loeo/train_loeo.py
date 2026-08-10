"""
SHARP++ 5-Class LOEO Protocol
=============================
Trains and evaluates the Residual CNN + CBAM + Bi-LSTM model using a
three-fold LOEO cross-validation with probability fusion across 4 antennas.

Classes: Empty, Sitting, Walking, Running, Jumping
Window: 340x100
Focal Loss (gamma=2)
"""

import argparse
import json
import os
import sys
import pickle
import numpy as np
import tensorflow as tf

# Import dataset_loader from current directory
from dataset_loader import CLASS_NAMES, prepare_dataset, compute_norm_stats, build_normalized_tf_dataset

# Import model from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build_sharp_plus_plus

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_THIS_DIR, 'outputs')

# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss Implementation
# ─────────────────────────────────────────────────────────────────────────────
# SparseFocalLoss removed in favor of built-in CategoricalFocalCrossentropy

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

# ─────────────────────────────────────────────────────────────────────────────
# Fold definitions
# ─────────────────────────────────────────────────────────────────────────────

LOEO_FOLDS = [
    {
        'fold': 1,
        'test_env':       'Bedroom',
        'train_envs':     ['Living Room', 'Laboratory'],
        'train_sessions': ['S6a', 'S6b', 'S7a'],
        'test_sessions':  ['S1a', 'S1b', 'S1c', 'S2a', 'S2b', 'S3a', 'S4a', 'S4b', 'S5a'],
    },
    {
        'fold': 2,
        'test_env':       'Living Room',
        'train_envs':     ['Bedroom', 'Laboratory'],
        'train_sessions': ['S1a', 'S1b', 'S1c', 'S2a', 'S2b', 'S3a', 'S4a', 'S4b', 'S5a', 'S7a'],
        'test_sessions':  ['S6a', 'S6b'],
    },
    {
        'fold': 3,
        'test_env':       'Laboratory',
        'train_envs':     ['Bedroom', 'Living Room'],
        'train_sessions': ['S1a', 'S1b', 'S1c', 'S2a', 'S2b', 'S3a', 'S4a', 'S4b', 'S5a', 'S6a', 'S6b'],
        'test_sessions':  ['S7a'],
    },
]

def filter_available(sessions, doppler_dir):
    available = [s for s in sessions if os.path.isdir(os.path.join(doppler_dir, s))]
    return available

def steps(n, batch_size):
    return max(1, int(np.ceil(n / batch_size)))

# ─────────────────────────────────────────────────────────────────────────────
# Single-fold runner
# ─────────────────────────────────────────────────────────────────────────────

def run_fold(fold_cfg, args):
    fold_n    = fold_cfg['fold']
    test_env  = fold_cfg['test_env']
    fold_dir  = os.path.join(args.output_dir, f'fold_{fold_n}')
    os.makedirs(fold_dir, exist_ok=True)

    num_classes = len(CLASS_NAMES)

    print(f"\n{'#'*70}")
    print(f"  FOLD {fold_n}  |  Test Environment: {test_env}")
    print(f"  Train envs : {fold_cfg['train_envs']}")
    print(f"{'#'*70}")

    train_sessions = filter_available(fold_cfg['train_sessions'], args.doppler_dir)
    test_sessions  = filter_available(fold_cfg['test_sessions'],  args.doppler_dir)

    if not train_sessions or not test_sessions:
        print(f"  [ERROR] Missing sessions for fold {fold_n}. Skipping.")
        return None

    # Load data
    print(f"\n[Fold {fold_n}] Loading training sessions into memory ...")
    (giant_tr, train_idx, val_idx, _, input_shape) = prepare_dataset(
        doppler_dir   = args.doppler_dir,
        session_list  = train_sessions,
        classes       = CLASS_NAMES,
        window_length = args.window,
        stride        = args.stride,
        train_ratio   = 0.75,
        val_ratio     = 0.25,
    )
    
    print(f"\n[Fold {fold_n}] Loading test sessions into memory ...")
    (giant_te, _, _, test_idx, _) = prepare_dataset(
        doppler_dir   = args.doppler_dir,
        session_list  = test_sessions,
        classes       = CLASS_NAMES,
        window_length = args.window,
        stride        = args.stride,
        train_ratio   = 0.0,
        val_ratio     = 0.0,
    )

    # Normalization (compute on training data only)
    print(f"\n[Fold {fold_n}] Computing normalization stats ...")
    mean, std = compute_norm_stats(giant_tr, train_idx, args.window, max_samples=5000)
    norm_path = os.path.join(fold_dir, 'norm_stats.pkl')
    with open(norm_path, 'wb') as fp:
        pickle.dump({'mean': mean, 'std': std}, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Build pipelines
    print(f"\n[Fold {fold_n}] Building tf.data pipelines ...")
    ds_train = build_normalized_tf_dataset(
        giant_tr, train_idx, args.window, args.batch_size, mean, std,
        mode='train', shuffle=True, repeat=True, augment=True, num_classes=num_classes
    )
    ds_val = build_normalized_tf_dataset(
        giant_tr, val_idx, args.window, args.batch_size, mean, std,
        mode='val', shuffle=False, repeat=True, num_classes=num_classes
    )
    
    # Test dataset yields (batch, 4, 340, 100, 1)
    ds_test = build_normalized_tf_dataset(
        giant_te, test_idx, args.window, args.batch_size, mean, std,
        mode='test', shuffle=False, repeat=False, num_classes=num_classes
    )

    steps_tr = steps(len(train_idx), args.batch_size)
    steps_va = steps(len(val_idx),   args.batch_size)

    # Build model
    print(f"\n[Fold {fold_n}] Building SHARP++ model ...")
    tf.keras.backend.clear_session()
    
    model = build_sharp_plus_plus(
        input_shape  = input_shape,
        num_classes  = num_classes,
        lstm_units   = args.lstm_units,
        dense_units  = args.dense_units,
        dropout_rate = args.dropout,
    )

    # Focal Loss (gamma=2)
    loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0)
    
    optimizer = get_optimizer(args.lr, args.weight_decay)

    model.compile(
        optimizer = optimizer,
        loss      = loss_fn,
        metrics   = [tf.keras.metrics.CategoricalAccuracy(name='accuracy')],
    )

    model_path = os.path.join(fold_dir, 'model.keras')
    callbacks  = [
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(3, args.patience // 3), min_lr=1e-6, verbose=1),
    ]

    print(f"\n[Fold {fold_n}] Training ...")
    history = model.fit(
        ds_train,
        epochs           = args.epochs,
        steps_per_epoch  = steps_tr,
        validation_data  = ds_val,
        validation_steps = steps_va,
        callbacks        = callbacks,
        verbose          = 1,
    )

    history_path = os.path.join(fold_dir, 'history.json')
    with open(history_path, 'w') as fp:
        json.dump({k: [float(v) for v in vs] for k, vs in history.history.items()}, fp, indent=2)

    # Evaluation with 4-antenna fusion
    print(f"\n[Fold {fold_n}] Evaluating on held-out {test_env} with Probability Fusion ...")
    
    all_preds_fused = []
    labels_true = []
    
    for x_batch, y_batch in ds_test:
        # x_batch shape: (batch_size, 4, 340, 100, 1)
        batch_size = tf.shape(x_batch)[0]
        
        # We need to run each antenna through the model and average probabilities
        p_final = tf.zeros((batch_size, num_classes), dtype=tf.float32)
        
        for ant in range(4):
            x_ant = x_batch[:, ant, :, :, :]  # (batch_size, 340, 100, 1)
            p_ant = model(x_ant, training=False)
            p_final += p_ant
            
        p_final = p_final / 4.0
        all_preds_fused.append(p_final.numpy())
        # y_batch is one-hot, need sparse indices for confusion matrix later
        labels_true.append(np.argmax(y_batch.numpy(), axis=-1))

    preds = np.concatenate(all_preds_fused, axis=0)
    labels_true = np.concatenate(labels_true, axis=0)

    # Temporal smoothing (11-samples)
    from scipy.ndimage import uniform_filter1d
    preds_smoothed = uniform_filter1d(preds, size=11, axis=0)
    labels_pred = np.argmax(preds_smoothed, axis=1)

    # Metrics
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
    idx = np.arange(num_classes)
    conf_matrix = confusion_matrix(labels_true, labels_pred, labels=idx)
    prec, rec, f1, supp = precision_recall_fscore_support(labels_true, labels_pred, labels=idx, zero_division=0)
    accuracy = accuracy_score(labels_true, labels_pred)

    print(f"\n{'='*70}")
    print(f"  Fold {fold_n} — Test: {test_env}")
    print(f"{'='*70}")
    print(f"  Overall Accuracy  : {accuracy:.4f}")
    print(f"  Macro Precision   : {np.mean(prec):.4f}")
    print(f"  Macro Recall      : {np.mean(rec):.4f}")
    print(f"  Macro F1-Score    : {np.mean(f1):.4f}")
    print(conf_matrix)
    print(f"{'='*70}")

    results = {
        'fold': fold_n,
        'test_env': test_env,
        'accuracy': float(accuracy),
        'precision': prec,
        'recall': rec,
        'fscore': f1,
        'macro_prec': float(np.mean(prec)),
        'macro_rec': float(np.mean(rec)),
        'macro_f1': float(np.mean(f1)),
        'conf_matrix': conf_matrix,
        'classes': CLASS_NAMES,
        'n_train': len(train_idx),
        'n_val': len(val_idx),
        'n_test': len(test_idx),
    }
    
    results_path = os.path.join(fold_dir, 'results.pkl')
    with open(results_path, 'wb') as fp:
        pickle.dump(results, fp)

    return results

def aggregate_and_print(fold_results, output_dir):
    completed = [r for r in fold_results if r is not None]
    if not completed: return

    accs = [r['accuracy'] for r in completed]
    f1s  = [r['macro_f1'] for r in completed]

    print(f"\n{'#'*70}")
    print(f"  5-Class LOEO Aggregate Results ({len(completed)} folds)")
    print(f"{'#'*70}")
    print(f"  Mean Accuracy : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    print(f"  Mean Macro F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    aggregate = {
        'n_folds': len(completed),
        'mean_accuracy': float(np.mean(accs)),
        'mean_macro_f1': float(np.mean(f1s)),
        'per_fold': [{'fold': r['fold'], 'test_env': r['test_env'], 'accuracy': r['accuracy'], 'macro_f1': r['macro_f1']} for r in completed],
    }
    with open(os.path.join(output_dir, 'aggregate_results.json'), 'w') as fp:
        json.dump(aggregate, fp, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--doppler_dir', default=os.path.join(os.path.dirname(_THIS_DIR), 'doppler_traces'))
    parser.add_argument('--window', type=int, default=340)
    parser.add_argument('--stride', type=int, default=17)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lstm_units', type=int, default=128)
    parser.add_argument('--dense_units', type=int, default=256)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output_dir', default=_OUTPUT_DIR)
    parser.add_argument('--folds', default='1,2,3')
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Disable GPU because Apple Silicon Metal plugin crashes with a NOT_FOUND 
            # error when trying to compile 1-channel Conv2D gradients with AdamW.
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError as e:
            print(e)
        # tf.keras.mixed_precision.set_global_policy('mixed_float16') # Disabled to fix Metal NOT_FOUND error on 1-channel inputs

    os.makedirs(args.output_dir, exist_ok=True)
    selected_folds = [int(f.strip()) for f in args.folds.split(',')]

    fold_results = []
    for fold_cfg in LOEO_FOLDS:
        if fold_cfg['fold'] not in selected_folds: continue
        result = run_fold(fold_cfg, args)
        fold_results.append(result)

    aggregate_and_print(fold_results, args.output_dir)

if __name__ == '__main__':
    main()
