"""
SHARP++ Evaluation / Test Script
=================================
Loads a saved SHARP++ model and evaluates it on any split of the SHARP dataset.

Usage
-----
    python test.py <doppler_dir> <subdirs> <activities> <model_path> [OPTIONS]

Positional arguments
    doppler_dir   Path to doppler_traces/ directory
    subdirs       Comma-separated session dirs (can differ from training)
    activities    Activity labels (must match training config)
    model_path    Path to the saved .keras model file

Optional arguments
    --batch_size  INT    Batch size              (default: 32)
    --split       STR    Data split to evaluate  (default: test)
                         choices: train | val | test | complete
    --output_dir  STR    Where to save result pickle  (default: ./outputs)
    --fix_paths          Re-map stored file paths to current doppler_dir

Example
-------
    # Evaluate on test set
    python test.py ../Python_code/doppler_traces/ S1a,S2a E,L,W,R,J \\
                   ./outputs/sharp_pp_E,L,W,R,J_network.keras

    # Evaluate on complete set with cross-session testing (train on S1*, test on S2*)
    python test.py ../Python_code/doppler_traces/ S2a,S2b E,L,W,R,J \\
                   ./outputs/sharp_pp_E,L,W,R,J_network.keras --split complete
"""

import argparse
import os
import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import (
    build_tf_dataset,
    load_sharp_file_lists,
    resolve_file_paths,
)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SHARP++ Evaluation Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('doppler_dir', help='Path to doppler_traces/ directory')
    parser.add_argument('subdirs',     help='Comma-separated session dirs, e.g. S1a,S2a')
    parser.add_argument('activities',  help='Activity labels, e.g. E,L,W,R,J')
    parser.add_argument('model_path',  help='Path to saved .keras model file')

    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--split',       type=str, default='test',
                        choices=['train', 'val', 'test', 'complete'],
                        help='Dataset split to evaluate')
    parser.add_argument('--output_dir',  type=str, default=os.getenv('OUTPUTS_DIR', './outputs'),
                        help='Directory to write result files')
    parser.add_argument('--fix_paths',   action='store_true',
                        help='Re-map stored file paths to current doppler_dir')
    args = parser.parse_args()

    session_list = args.subdirs.split(',')
    activity_str = args.activities
    activities   = activity_str.split(',')
    num_classes  = len(activities)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── GPU setup ─────────────────────────────────────────────────────────────
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)
    model.summary(line_length=80)

    # Infer input shape from the model itself
    input_shape = tuple(model.input_shape[1:])   # strip batch dim
    print(f"[INFO] Model input shape: {input_shape}")

    # ── Load file lists ───────────────────────────────────────────────────────
    print(f"[INFO] Loading '{args.split}' split for sessions: {session_list}")
    files, labels = load_sharp_file_lists(
        args.doppler_dir, session_list, activity_str, args.split)

    if args.fix_paths:
        files = resolve_file_paths(files, args.doppler_dir)

    print(f"[INFO] Samples: {len(files)}")

    # Label distribution
    unique, counts = np.unique(labels, return_counts=True)
    dist = {activities[int(u)]: int(c) for u, c in zip(unique, counts)}
    print(f"[INFO] Class distribution: {dist}")

    # ── Build dataset ─────────────────────────────────────────────────────────
    cache_prefix = os.path.join(args.output_dir, f'eval_cache_{args.split}')
    for ext in ['.data-00000-of-00001', '.index']:
        if os.path.exists(cache_prefix + ext):
            os.remove(cache_prefix + ext)

    dataset = build_tf_dataset(
        files, labels, input_shape, args.batch_size,
        shuffle   = False,
        cache_file= None,     # no caching for evaluation
        prefetch  = True,
        repeat    = False,
    )

    # ── Predict ───────────────────────────────────────────────────────────────
    print("[INFO] Running inference ...")
    labels_true  = np.array(labels)
    predictions  = model.predict(dataset, verbose=1)
    predictions  = predictions[:labels_true.shape[0]]     # clip any batch padding
    labels_pred  = np.argmax(predictions, axis=1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    from sklearn.metrics import (
        confusion_matrix, precision_recall_fscore_support, accuracy_score
    )

    labels_considered = np.arange(num_classes)
    conf_matrix       = confusion_matrix(labels_true, labels_pred, labels=labels_considered)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        labels_true, labels_pred, labels=labels_considered, zero_division=0)
    accuracy = accuracy_score(labels_true, labels_pred)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SHARP++ — Evaluation Results")
    print(f"{'='*65}")
    print(f"  Split     : {args.split.upper()}")
    print(f"  Sessions  : {session_list}")
    print(f"  Model     : {os.path.basename(args.model_path)}")
    print(f"{'─'*65}")
    print(f"  Overall Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Macro Precision  : {np.mean(precision):.4f}")
    print(f"  Macro Recall     : {np.mean(recall):.4f}")
    print(f"  Macro F1-Score   : {np.mean(fscore):.4f}")
    print(f"{'─'*65}")
    print(f"  {'Activity':<14} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'Count':>7}")
    print(f"{'─'*65}")
    for i, act in enumerate(activities):
        n = int(np.sum(labels_true == i))
        print(f"  {act:<14} {precision[i]:>10.4f} {recall[i]:>8.4f} {fscore[i]:>10.4f} {n:>7d}")
    print(f"{'─'*65}")
    print(f"\n  Confusion Matrix  (rows=true, cols=pred)")
    print(f"  Labels: {activities}")
    print(conf_matrix)
    print(f"{'='*65}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'conf_matrix'    : conf_matrix,
        'accuracy'       : float(accuracy),
        'precision'      : precision,
        'recall'         : recall,
        'fscore'         : fscore,
        'activities'     : activities,
        'split'          : args.split,
        'sessions'       : session_list,
        'model_path'     : args.model_path,
        'input_shape'    : input_shape,
    }
    result_file = os.path.join(
        args.output_dir,
        f"results_{args.split}_{activity_str}_{args.subdirs.replace(',', '_')}.txt"
    )
    with open(result_file, 'wb') as fp:
        pickle.dump(results, fp)
    print(f"\n[INFO] Results saved → {result_file}")


if __name__ == '__main__':
    main()
