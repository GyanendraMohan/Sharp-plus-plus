import json, os, sys

OUT_DIR = sys.argv[1]

# ─────────────────────────────── shared cells ───────────────────────────────

SETUP = """
import os, glob, pickle, itertools, shutil, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import uniform_filter1d
from scipy.special import softmax
from sklearn.metrics import recall_score, f1_score, confusion_matrix


def find_root(pattern):
    hits = glob.glob(f'/kaggle/input/**/{pattern}', recursive=True)
    return os.path.dirname(os.path.dirname(hits[0])) if hits else None


DOPPLER_DIR = find_root('S1a/S1a_E_stream_0.txt')
OUT = '/kaggle/working/results'
os.makedirs(OUT, exist_ok=True)
GPUS = tf.config.list_physical_devices('GPU')
for gpu in GPUS:
    tf.config.experimental.set_memory_growth(gpu, True)

WINDOW, NUM_ANTENNAS, BATCH_SIZE, EPOCHS = 340, 4, 32, 25
# each model trains with its own recipe; both are tested on the same windows
RECIPES = {'Doppler-ResAT': {'stride': 17, 'logits': False},  # ISED configuration, unchanged
           'SHARP': {'stride': 30, 'logits': True}}           # SHARP's CSI_doppler_create_dataset_train.py + CSI_network.py
MODELS = list(RECIPES)
ABLATIONS = {}  # Doppler-ResAT variants with a layer removed: {name: build_sharp_plus_plus options}
CLASSES = {5: ['E', 'L', 'W', 'R', 'J'], 8: ['E', 'L', 'S', 'C', 'W', 'R', 'J', 'H']}
ISED_RECORDINGS = ['E', 'L', 'W', 'R', 'J', 'J1', 'J2']  # recordings the ISED LOEO runs used
SMOOTH_ACROSS_RECORDINGS = True  # as the ISED fold scripts; False smooths each recording separately
BEDROOM = ['S1a', 'S1b', 'S1c', 'S2a', 'S2b', 'S3a', 'S4a', 'S4b', 'S5a']
LOEO = {1: (['S6a', 'S6b', 'S7a'], BEDROOM), 2: (BEDROOM + ['S7a'], ['S6a', 'S6b'])}  # (train, test)
"""

MODELS_MD = "## Models: Doppler-ResAT (ISED) and SHARP (`network_utility.py`)"

MODELS_CODE = """
def residual_block(x, filters, name_prefix='res'):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, name=f'{name_prefix}_conv1')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = tf.keras.layers.ReLU(name=f'{name_prefix}_relu1')(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, name=f'{name_prefix}_conv2')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, (1, 1), padding='same', use_bias=False,
                                          name=f'{name_prefix}_proj_conv')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f'{name_prefix}_proj_bn')(shortcut)
    x = tf.keras.layers.Add(name=f'{name_prefix}_add')([x, shortcut])
    return tf.keras.layers.ReLU(name=f'{name_prefix}_relu2')(x)


def channel_attention(x, reduction_ratio=8, name_prefix='ca'):
    channels = x.shape[-1]
    dense_down = tf.keras.layers.Dense(max(1, channels // reduction_ratio), activation='relu', use_bias=False,
                                       name=f'{name_prefix}_dense_down')
    dense_up = tf.keras.layers.Dense(channels, use_bias=False, name=f'{name_prefix}_dense_up')
    avg_pool = tf.keras.layers.GlobalAveragePooling2D(name=f'{name_prefix}_avg_pool')(x)
    avg_pool = tf.keras.layers.Reshape((1, 1, channels), name=f'{name_prefix}_avg_reshape')(avg_pool)
    max_pool = tf.keras.layers.GlobalMaxPooling2D(name=f'{name_prefix}_max_pool')(x)
    max_pool = tf.keras.layers.Reshape((1, 1, channels), name=f'{name_prefix}_max_reshape')(max_pool)
    scale = tf.keras.layers.Activation('sigmoid', name=f'{name_prefix}_sigmoid')(
        dense_up(dense_down(avg_pool)) + dense_up(dense_down(max_pool)))
    return tf.keras.layers.Multiply(name=f'{name_prefix}_mul')([x, scale])


def spatial_attention(x, kernel_size=7, name_prefix='sa'):
    avg_pool = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                                      name=f'{name_prefix}_avg_pool')(x)
    max_pool = tf.keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                                      name=f'{name_prefix}_max_pool')(x)
    concat = tf.keras.layers.Concatenate(axis=-1, name=f'{name_prefix}_concat')([avg_pool, max_pool])
    attn_map = tf.keras.layers.Conv2D(1, kernel_size, padding='same', use_bias=False, name=f'{name_prefix}_conv')(concat)
    attn_map = tf.keras.layers.Activation('sigmoid', name=f'{name_prefix}_sigmoid')(attn_map)
    return tf.keras.layers.Multiply(name=f'{name_prefix}_mul')([x, attn_map])


def cbam_block(x, reduction_ratio=8, spatial_kernel=7, name_prefix='cbam'):
    x = channel_attention(x, reduction_ratio, name_prefix=f'{name_prefix}_ch')
    return spatial_attention(x, spatial_kernel, name_prefix=f'{name_prefix}_sp')


def build_sharp_plus_plus(input_shape, num_classes, lstm_units=128, dense_units=256, dropout_rate=0.5,
                          use_cbam=True, use_bilstm=True):
    # use_cbam / use_bilstm=False remove that layer for the ablation; the defaults are the ISED model
    inputs = tf.keras.Input(shape=input_shape, name='spectrogram_input')
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', use_bias=False, name='stem_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.ReLU(name='stem_relu')(x)
    x = residual_block(x, 64, name_prefix='res1')
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = residual_block(x, 128, name_prefix='res2')
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = residual_block(x, 256, name_prefix='res3')
    if use_cbam:
        x = cbam_block(x, reduction_ratio=8, spatial_kernel=7)
    if use_bilstm:
        x = tf.keras.layers.Reshape((-1, x.shape[2] * x.shape[3]), name='reshape_lstm')(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True, dropout=0.2,
                                                               recurrent_dropout=0.0), name='bilstm_1')(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True, dropout=0.2,
                                                               recurrent_dropout=0.0), name='bilstm_2')(x)
        x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    x = tf.keras.layers.Dense(dense_units, activation='relu', name='dense_embed')(x)
    x = tf.keras.layers.Dropout(dropout_rate, name='dropout')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype='float32', name='output')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='SHARP_plus_plus')


def get_optimizer(lr, weight_decay):
    try:
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    except AttributeError:
        return tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=weight_decay)


def conv2d_bn(x_in, filters, kernel_size, strides=(1, 1), padding='same', activation='relu', bn=False, name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, name=name)(x_in)
    if bn:
        bn_name = None if name is None else name + '_bn'
        x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name)(x)
    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)
    return x


def reduction_a_block_small(x_in, base_name):
    x1 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='valid')(x_in)
    x2 = conv2d_bn(x_in, 5, (2, 2), strides=(2, 2), padding='valid', name=base_name + 'conv2_1_res_a')
    x3 = conv2d_bn(x_in, 3, (1, 1), name=base_name + 'conv3_1_res_a')
    x3 = conv2d_bn(x3, 6, (2, 2), name=base_name + 'conv3_2_res_a')
    x3 = conv2d_bn(x3, 9, (4, 4), strides=(2, 2), padding='same', name=base_name + 'conv3_3_res_a')
    return tf.keras.layers.Concatenate()([x1, x2, x3])


def csi_network_inc_res(input_sh, output_sh):
    x_input = tf.keras.Input(input_sh)
    x2 = reduction_a_block_small(x_input, base_name='1st')
    x3 = conv2d_bn(x2, 3, (1, 1), name='conv4')
    x = tf.keras.layers.Flatten()(x3)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_sh, activation=None, name='dense2')(x)
    return tf.keras.Model(inputs=x_input, outputs=x, name='csi_model')
"""

DATA_MD = "## Data, training and inference"

DATA_CODE = """
def recordings(root, session, classes, ised=False):
    # (label, name) of each recording; the label is the first letter of the activity, as in SHARP's dataset scripts
    names = sorted({f[len(session) + 1:].split('_stream_')[0] for f in os.listdir(f'{root}/{session}') if '_stream_' in f})
    if ised:  # the ISED fold scripts' order: class by class, then J, J1, J2
        return [(classes.index(c), n) for c in classes for n in ISED_RECORDINGS if n[0] == c and n in names]
    return [(classes.index(n[0]), n) for n in names if n[0] in classes]


def load_recording(root, session, name):
    # (time, 100, antennas), mean-subtracted along time
    streams = [pickle.load(open(f'{root}/{session}/{session}_{name}_stream_{a}.txt', 'rb')) for a in range(NUM_ANTENNAS)]
    return np.stack([s - s.mean(axis=0, keepdims=True) for s in streams], axis=-1).astype(np.float16)


def load_sessions(root, sessions, classes, ised=False):
    # all recordings in one array, plus (offset, length, label) of each recording
    data, recs, offset = [], [], 0
    for s in sessions:
        for label, name in recordings(root, s, classes, ised):
            x = load_recording(root, s, name)
            data.append(x)
            recs.append((offset, len(x), label))
            offset += len(x)
    return np.concatenate(data), recs


def window_index(recs, lo, hi, stride):
    # (start, label, antenna) of the windows inside the time split [lo, hi) of each recording
    return np.array([(o + t, label, ant) for o, T, label in recs for t in range(int(T * lo), int(T * hi) - WINDOW + 1, stride)
                     for ant in range(NUM_ANTENNAS)], dtype=np.int32).reshape(-1, 3)


def compute_norm_stats(giant_array, indices, window_length, max_samples=5000):
    n = min(max_samples, len(indices))
    sample_idxs = indices[np.random.choice(len(indices), n, replace=False)]
    samples = np.stack([giant_array[start:start + window_length, :, ant:ant + 1] for (start, _, ant) in sample_idxs])
    samples = samples.astype(np.float64)
    mean = samples.mean(axis=(0, 1), keepdims=True)[0].astype(np.float32)
    std = np.maximum(samples.std(axis=(0, 1), keepdims=True), 1e-8)[0].astype(np.float32)
    return mean, std


def make_dataset(master, idx, mean, std, num_classes, shuffle, augment):
    master_t, mean_t, std_t = tf.constant(master), tf.constant(mean), tf.constant(std)
    ds = tf.data.Dataset.from_tensor_slices(idx)
    if shuffle:
        ds = ds.shuffle(min(len(idx), 20000), reshuffle_each_iteration=True)
    ds = ds.repeat()

    def window(i):
        x = (tf.cast(master_t[i[0]:i[0] + WINDOW, :, i[2]:i[2] + 1], tf.float32) - mean_t) / std_t
        return tf.ensure_shape(x, (WINDOW, master.shape[1], 1)), tf.one_hot(i[1], depth=num_classes)

    def augment_fn(x, label):
        x = x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.05)

        def mask_time():
            win_len = tf.shape(x)[0]
            mask_size = tf.random.uniform([], 1, 30, dtype=tf.int32)
            mask_start = tf.random.uniform([], 0, win_len - mask_size, dtype=tf.int32)
            t = tf.range(win_len)
            mask = tf.logical_or(t < mask_start, t >= mask_start + mask_size)
            return x * tf.reshape(tf.cast(mask, tf.float32), [win_len, 1, 1])
        return tf.cond(tf.random.uniform([]) < 0.3, mask_time, lambda: x), label

    ds = ds.map(window, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def set_policy(name):
    use_mixed = name != 'SHARP' and GPUS  # ISED trained in mixed precision; SHARP in float32
    tf.keras.mixed_precision.set_global_policy('mixed_float16' if use_mixed else 'float32')


class Resume(tf.keras.callbacks.Callback):
    # saves the training state after every epoch and stops before the deadline; a later session continues from it
    KEYS = ['best', 'wait', 'best_epoch', 'best_weights', 'cooldown_counter']  # EarlyStopping, ReduceLROnPlateau

    def __init__(self, path, others, deadline, mean, std, state):
        super().__init__()
        self.path, self.others, self.deadline, self.mean, self.std, self.state = path, others, deadline, mean, std, state
        self.interrupted = False

    def on_train_begin(self, logs=None):
        # runs after the other callbacks have reset themselves
        if self.state:
            for cb, attrs in zip(self.others, self.state['callbacks']):
                for k, v in attrs.items():
                    setattr(cb, k, v)
        self.t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        state = {'epoch': epoch + 1, 'weights': self.model.get_weights(), 'optimizer': [v.numpy() for v in opt.variables],
                 'learning_rate': float(np.array(opt.learning_rate)), 'mean': self.mean, 'std': self.std,
                 'callbacks': [{k: getattr(cb, k) for k in self.KEYS if hasattr(cb, k)} for cb in self.others]}
        with open(self.path, 'wb') as fp:
            pickle.dump(state, fp)
        now = time.time()
        if not self.model.stop_training and epoch + 1 < EPOCHS and now + 1.2 * (now - self.t) > self.deadline:
            self.model.stop_training = self.interrupted = True
            print(f'Epoch {epoch + 1}: stopping before the time limit; the training state is saved to {self.path}')
        self.t = now


def train_model(name, master, recs, splits, num_classes, seed, resume=None):
    # one model, trained with its own recipe on the (train, val) time splits of the recordings;
    # resume=(file, deadline) saves the state every epoch, stops before the deadline and continues from an attached file
    train_idx, val_idx = (window_index(recs, lo, hi, RECIPES[name]['stride']) for lo, hi in splits)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    set_policy(name)
    state = find_result(resume[0]) if resume else None
    if name == 'SHARP':
        mean, std = np.zeros((1, master.shape[1], 1), np.float32), np.ones((1, master.shape[1], 1), np.float32)
        model = csi_network_inc_res((WINDOW, master.shape[1], 1), num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
    else:
        mean, std = (state['mean'], state['std']) if state else compute_norm_stats(master, train_idx, WINDOW)
        model = build_sharp_plus_plus((WINDOW, master.shape[1], 1), num_classes, **ABLATIONS.get(name, {}))
        model.compile(optimizer=get_optimizer(1e-4, 1e-4), loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0),
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)]
    start = 0
    if state:
        model.set_weights(state['weights'])
        model.optimizer.build(model.trainable_variables)
        if len(model.optimizer.variables) != len(state['optimizer']):
            raise ValueError(f'{resume[0]} does not match this model')
        for v, value in zip(model.optimizer.variables, state['optimizer']):
            v.assign(value)
        model.optimizer.learning_rate = state['learning_rate']
        start = state['epoch']
        print(f'{name}: continuing from epoch {start} of {resume[0]}')
    if resume:
        callbacks.append(Resume(f'{OUT}/{resume[0]}', list(callbacks), resume[1], mean, std, state))
    history = model.fit(make_dataset(master, train_idx, mean, std, num_classes, True, name != 'SHARP'),
                        epochs=EPOCHS, initial_epoch=start, steps_per_epoch=int(np.ceil(len(train_idx) / BATCH_SIZE)),
                        validation_data=make_dataset(master, val_idx, mean, std, num_classes, False, False),
                        validation_steps=int(np.ceil(len(val_idx) / BATCH_SIZE)), callbacks=callbacks, verbose=2)
    if resume and callbacks[-1].interrupted:
        return None
    if resume:
        os.remove(f'{OUT}/{resume[0]}')
    # Doppler-ResAT restores its best epoch; SHARP keeps its last epoch, as CSI_network.py does
    epoch = callbacks[0].best_epoch + 1 if name != 'SHARP' else start + len(history.history['loss'])
    return model, mean, std, epoch


def predict_recording(model, x, lo, stride, mean, std):
    # per-antenna outputs (windows, antennas, classes) from fraction lo to the end of one recording
    starts = range(int(len(x) * lo), len(x) - WINDOW + 1, stride)
    w = (np.stack([x[t:t + WINDOW] for t in starts]).astype(np.float32) - mean) / std
    w = np.moveaxis(w, -1, 1).reshape(-1, WINDOW, x.shape[1], 1)
    return model.predict(w, batch_size=4 * BATCH_SIZE, verbose=0).reshape(len(starts), NUM_ANTENNAS, -1)


def predict_sets(model, root, sets, classes, lo, stride, mean, std, ised=False):
    # {set: [(outputs, labels) for each recording]}; lo gives the start fraction per set (default 0)
    out = {}
    for name, sessions in sets.items():
        out[name] = []
        for s in sessions:
            for label, rec in recordings(root, s, classes, ised):
                P = predict_recording(model, load_recording(root, s, rec), lo.get(name, 0), stride, mean, std)
                out[name].append((P, np.full(len(P), label)))
    return out


def find_result(fname):
    # results of earlier sessions can be attached as a Kaggle dataset; they are copied into OUT
    if not os.path.exists(f'{OUT}/{fname}'):
        hits = glob.glob(f'/kaggle/input/**/{fname}', recursive=True)
        if not hits:
            return None
        shutil.copy(hits[0], OUT)
    with open(f'{OUT}/{fname}', 'rb') as fp:
        return pickle.load(fp)


def save_result(fname, result):
    with open(f'{OUT}/{fname}', 'wb') as fp:
        pickle.dump(result, fp)
"""

METRICS_MD = "## Metrics (protocol code; smoothing as in the ISED fold scripts; SHARP's vote fusion)"

METRICS_CODE = """
def smooth(p, k=11):
    # p: (T, C) fused probabilities in time order
    return uniform_filter1d(p, size=k, axis=0)


def sharp_vote(Z):
    # SHARP's decision fusion (CSI_network_test.py): majority of the antenna decisions, summed logits break ties
    pred = Z.sum(axis=1).argmax(axis=1)
    for i, v in enumerate(Z.argmax(axis=2)):
        lab, cnt = np.unique(v, return_counts=True)
        if len(lab) == 1 or (len(lab) == 2 and cnt[0] != cnt[1]):
            pred[i] = lab[cnt.argmax()]
    return pred


def evaluate(streams, classes, antennas=(0, 1, 2, 3), use_smoothing=True, logits=False, vote=False):
    # streams: list of (P, y); P: (T, 4, C) per-antenna softmax (logits for SHARP), y: (T,) labels
    y_true = np.concatenate([y for _, y in streams])
    if vote:
        y_pred = np.concatenate([sharp_vote(P) for P, _ in streams])
    else:
        fused = [(softmax(P, axis=-1) if logits else P)[:, list(antennas), :].mean(axis=1) for P, _ in streams]
        if use_smoothing:
            fused = [smooth(np.concatenate(fused))] if SMOOTH_ACROSS_RECORDINGS else [smooth(f) for f in fused]
        y_pred = np.concatenate(fused).argmax(axis=1)
    labels = list(range(len(classes)))
    present = [c for c in labels if c in set(y_true)]  # S2 has no S, C or H recordings
    rec = recall_score(y_true, y_pred, labels=present, average=None) * 100
    return {
        "per_class_recall": dict(zip([classes[c] for c in present], np.round(rec, 2))),
        "mean_recall": round(float(rec.mean()), 2),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=present, average="macro")), 3),
        "accuracy": round(float(100 * (y_true == y_pred).mean()), 2),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
    }


def flat(e):
    return {'mean_recall': e['mean_recall'], 'macro_f1': e['macro_f1'], 'accuracy': e['accuracy'],
            **{f'recall_{c}': v for c, v in e['per_class_recall'].items()}}


def antenna_subsets(streams, classes, n, **kw):
    # {metric: (mean, std)} over all combinations of n antennas
    runs = [flat(evaluate(streams, classes, s, **kw)) for s in itertools.combinations(range(NUM_ANTENNAS), n)]
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in runs[0]}


def confusion_frame(cm, classes, **tags):
    # raw window counts, rows = true class, columns = predicted class
    df = pd.DataFrame(cm, index=pd.Index(classes, name='true'), columns=classes).reset_index()
    return df.assign(**tags)[list(tags) + list(df.columns)]


# rows reported per model: (model, label, fusion)
VARIANTS = [('Doppler-ResAT', 'Doppler-ResAT', {}),
            ('SHARP', 'SHARP re-run (probability fusion)', {}),
            ('SHARP', 'SHARP re-run (vote fusion, no smoothing)', {'vote': True})]
"""

# ───────────────────────── Experiments 1 and 2 ─────────────────────────

SHARP_PAPER = {
    1: """
# SHARP Table 3 (decision fusion): per-class recall (%), mean recall, macro-F1
SHARP_PAPER = {
    'S1': [100, 100, 100, 100, 97.93, 99.59, 0.996], 'S2': [100, 100, 100, 99.28, 99.64, 99.78, 0.998],
    'S3': [100, 100, 96.64, 100, 100, 99.33, 0.993], 'S4': [100, 100, 99.84, 93.27, 94.63, 97.55, 0.976],
    'S5': [100, 100, 98.58, 84.93, 99.45, 96.59, 0.966], 'S6': [99.76, 100, 100, 100, 99.20, 99.79, 0.998],
    'S7': [100, 100, 81.24, 98.71, 100, 95.99, 0.960],
}
""",
    2: """
# SHARP Table 6: per-class recall (%), mean recall, macro-F1
SHARP_PAPER = {
    'S1': [100, 100, 100, 100, 100, 100, 100, 96.02, 99.50, 0.994],
    'S2': [100, 100, 100, 100, 100, 99.84, 100, 94.56, 99.30, 0.993],
    'S6': [100, 100, 100, 100, 100, 73.16, 99.82, 90.22, 95.40, 0.953],
}
""",
}

EXP12_CONFIG = {
    1: """
SEEDS = [0, 1, 2]
NUM_CLASSES = 5
STRIDE_TEST = 30
S1 = ['S1a', 'S1b', 'S1c']
TEST_SETS = {'S1': S1, 'S2': ['S2a', 'S2b'], 'S3': ['S3a'], 'S4': ['S4a', 'S4b'], 'S5': ['S5a'],
             'S6': ['S6a', 'S6b'], 'S7': ['S7a']}
KEY_SETS = ['S4', 'S5', 'S7']  # confusion matrices to show
""",
    2: """
SEEDS = [0, 1, 2]
NUM_CLASSES = 8
STRIDE_TEST = 30
S1 = ['S1a', 'S1b', 'S1c']
TEST_SETS = {'S1': S1, 'S2': ['S2a', 'S2b'], 'S6': ['S6a', 'S6b']}
KEY_SETS = ['S6']  # confusion matrices to show
""",
}

EXP12_RUN = """
classes = CLASSES[NUM_CLASSES]
master, recs = load_sessions(DOPPLER_DIR, S1, classes)
for seed in SEEDS:
    for name in MODELS:
        fname = f'exp{EXP}_{name}_seed{seed}.pkl'
        if find_result(fname):
            continue
        model, mean, std, epoch = train_model(name, master, recs, [(0, 0.6), (0.6, 0.8)], len(classes), seed)
        model.save_weights(f'{OUT}/exp{EXP}_{name}_seed{seed}.weights.h5')
        probs = predict_sets(model, DOPPLER_DIR, TEST_SETS, classes, {'S1': 0.8}, STRIDE_TEST, mean, std)
        save_result(fname, {'model': name, 'seed': seed, 'probs': probs, 'logits': RECIPES[name]['logits'],
                            'epoch': epoch, 'train_stride': RECIPES[name]['stride'], 'test_stride': STRIDE_TEST})
del master
"""

EXP12_REPORT = """
runs = {name: [r for r in (find_result(f'exp{EXP}_{name}_seed{s}.pkl') for s in SEEDS) if r] for name in MODELS}
main, per_seed, subsets, confusions, windows = [], [], [], [], []
for set_name in TEST_SETS:
    paper = SHARP_PAPER[set_name]
    main.append({'Set': set_name, 'Model': 'SHARP (paper)', **dict(zip(classes, paper)),
                 'mean_recall': paper[-2], 'macro_f1': paper[-1]})
    for name, label, kw in VARIANTS:
        if not runs[name]:
            continue
        sm = [evaluate(r['probs'][set_name], classes, logits=r['logits'], **kw) for r in runs[name]]
        raw = [evaluate(r['probs'][set_name], classes, logits=r['logits'], use_smoothing=False, **kw) for r in runs[name]]
        row = {'Set': set_name, 'Model': label, 'Seeds': len(sm)}
        for c in sm[0]['per_class_recall']:
            row[c] = f"{np.mean([e['per_class_recall'][c] for e in sm]):.2f} [{np.mean([e['per_class_recall'][c] for e in raw]):.2f}]"
        for k in ('mean_recall', 'macro_f1', 'accuracy'):
            row[k] = f"{np.mean([e[k] for e in sm]):.3f} ± {np.std([e[k] for e in sm]):.3f} [{np.mean([e[k] for e in raw]):.3f}]"
        main.append(row)
        for r, a, b in zip(runs[name], sm, raw):
            per_seed.append({'Set': set_name, 'Model': label, 'Seed': r['seed'], 'Epoch': r['epoch'], **flat(a),
                             **{f'{k} [no smoothing]': v for k, v in flat(b).items()}})
        confusions.append(confusion_frame(sum(e['confusion'] for e in sm), classes, Set=set_name, Model=label,
                                          Seeds=len(sm)))
        if not kw:
            for n in range(1, NUM_ANTENNAS + 1):
                for use_smoothing in (True, False):
                    m = [antenna_subsets(r['probs'][set_name], classes, n, use_smoothing=use_smoothing,
                                         logits=r['logits'])['mean_recall'][0] for r in runs[name]]
                    subsets.append({'Set': set_name, 'Model': label, 'Smoothing': use_smoothing, 'Antennas': n,
                                    'Mean recall': round(float(np.mean(m)), 2), 'Std over seeds': round(float(np.std(m)), 2)})
    y = np.concatenate([y for _, y in next(iter(r for v in runs.values() for r in v))['probs'][set_name]])
    windows.append({'Set': set_name, 'Test stride': STRIDE_TEST, **dict(zip(classes, np.bincount(y, minlength=len(classes))))})

tables = {'main': pd.DataFrame(main)[['Set', 'Model', 'Seeds', *classes, 'mean_recall', 'macro_f1', 'accuracy']], 'per_seed': pd.DataFrame(per_seed), 'antenna_subsets': pd.DataFrame(subsets),
          'confusion': pd.concat(confusions), 'test_windows': pd.DataFrame(windows),
          'epochs': pd.DataFrame([{'Model': r['model'], 'Seed': r['seed'], 'Epoch used': r['epoch'],
                                   'Train stride': r['train_stride']} for v in runs.values() for r in v])}
for key, df in tables.items():
    df.to_csv(f'{OUT}/exp{EXP}_{key}.csv', index=False)

print('Mean over seeds with smoothing, no smoothing in [brackets]; SHARP (paper) copied from the SHARP article')
display(tables['main'])
display(tables['antenna_subsets'].pivot_table(index=['Set', 'Model', 'Smoothing'], columns='Antennas', values='Mean recall'))
for set_name in KEY_SETS:
    print(f'Confusion matrices, {set_name} (windows summed over seeds)')
    display(tables['confusion'][tables['confusion']['Set'] == set_name])
display(tables['test_windows'], tables['epochs'])
"""

# ───────────────────────── LOEO comparison ─────────────────────────

LOEO_CODE = """
STRIDE_LOEO_TEST = 17  # ISED test stride
DEADLINE = time.time() + 11.25 * 3600  # Kaggle stops a run at 12 h; training is saved and stopped before that


def run_loeo(fold, name):
    # one model on one LOEO fold; the trained ISED fold model is reused when attached
    fname = f'loeo_{name}_fold{fold}.pkl'
    if find_result(fname):
        return
    classes = CLASSES[5]
    train_sessions, test_sessions = LOEO[fold]
    ised = glob.glob(f'/kaggle/input/**/fold_{fold}/model.keras', recursive=True)
    if name == 'Doppler-ResAT' and ised:
        tf.keras.backend.clear_session()
        set_policy(name)
        model = build_sharp_plus_plus((WINDOW, 100, 1), len(classes))
        model.load_weights(ised[0])
        with open(os.path.join(os.path.dirname(ised[0]), 'norm_stats.pkl'), 'rb') as fp:
            norm = pickle.load(fp)
        mean, std, epoch = norm['mean'], norm['std'], 'ISED model'
    else:
        master, recs = load_sessions(DOPPLER_DIR, train_sessions, classes, ised=True)
        trained = train_model(name, master, recs, [(0, 0.75), (0.75, 1)], len(classes), SEED,
                              resume=(f'{fname[:-4]}.resume.pkl', DEADLINE))
        del master
        if trained is None:
            print(f'{name}, fold {fold}: not finished in this session. Run the notebook again with this output attached.')
            return
        model, mean, std, epoch = trained
        model.save_weights(f'{OUT}/{fname[:-4]}.weights.h5')
    probs = predict_sets(model, DOPPLER_DIR, {'test': test_sessions}, classes, {}, STRIDE_LOEO_TEST, mean, std,
                         ised=True)['test']
    save_result(fname, {'model': name, 'fold': fold, 'probs': probs, 'logits': RECIPES[name]['logits'],
                        'epoch': epoch, 'test_stride': STRIDE_LOEO_TEST})
"""

LOEO_CONFIG = """
FOLDS = [1, 2]
SEED = 0
HELD_OUT = {1: 'bedroom', 2: 'living room'}
CLASS_NAMES = ['Empty', 'Sitting', 'Walking', 'Running', 'Jumping']
"""

LOEO_RUN = """
for name in ['SHARP', 'Doppler-ResAT']:  # Doppler-ResAT fold 2 (~17 h) runs last and continues over two sessions
    for fold in FOLDS:
        run_loeo(fold, name)
"""

LOEO_REPORT = """
classes = CLASSES[5]
rows, confusions, windows = [], [], []
for fold in FOLDS:
    for name, label, kw in VARIANTS:
        r = find_result(f'loeo_{name}_fold{fold}.pkl')
        if r is None:
            continue
        e = evaluate(r['probs'], classes, logits=r['logits'], **kw)
        raw = evaluate(r['probs'], classes, logits=r['logits'], use_smoothing=False, **kw)
        rows.append({'Fold': fold, 'Held out': HELD_OUT[fold], 'Model': label, 'Epoch used': r['epoch'], **flat(e),
                     **{f'{k} [no smoothing]': v for k, v in flat(raw).items()}})
        confusions.append(confusion_frame(e['confusion'], classes, Fold=fold, Model=label))
        if name == 'Doppler-ResAT':
            y = np.concatenate([y for _, y in r['probs']])
            windows.append({'Fold': fold, 'Test stride': r['test_stride'], **dict(zip(classes, np.bincount(y, minlength=5)))})

tables = {'comparison': pd.DataFrame(rows), 'confusion': pd.concat(confusions), 'test_windows': pd.DataFrame(windows)}
for key, df in tables.items():
    df.to_csv(f'{OUT}/loeo_{key}.csv', index=False)
print('4-antenna fusion with smoothing (no smoothing in the [no smoothing] columns); recall and accuracy in %')
for df in tables.values():
    display(df)

# per-class recall in the held-out rooms
results = tables['comparison']
fig, axes = plt.subplots(1, len(FOLDS), figsize=(6 * len(FOLDS), 3.5), squeeze=False, sharey=True)
for ax, fold in zip(axes[0], FOLDS):
    r = results[results['Fold'] == fold].reset_index(drop=True)
    width = 0.8 / len(r)
    for k, row in r.iterrows():
        ax.bar(np.arange(len(classes)) + (k - (len(r) - 1) / 2) * width, [row[f'recall_{c}'] for c in classes], width,
               label=row['Model'])
    ax.set(xticks=range(len(classes)), xticklabels=CLASS_NAMES, ylim=(0, 100), title=f'Fold {fold}: {HELD_OUT[fold]} held out')
axes[0][0].set_ylabel('Recall (%)')
axes[0][0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/loeo_per_class_recall.png', dpi=200)
fig.savefig(f'{OUT}/loeo_per_class_recall.pdf')
plt.show()
"""

EXP4_CONFIG = """
FOLDS = [1, 2]
D_LIST = [1, 2, 4, 8]  # D = 1 reuses Experiment 3's results when they are attached
SEED = 0
RATE, WAVELENGTH = 173, 0.0576  # packets/s at full rate; metres at 5.21 GHz
"""

EXP4_DOPPLER = """
CFR_DIR = find_root('S1a/S1a_E_stream_0.mat')
if CFR_DIR is None and any(D > 1 for D in D_LIST):
    raise FileNotFoundError('Experiment 4 needs the sanitized CFR .mat files (S1a/S1a_E_stream_0.mat, ...). Packets must be '
                            'dropped before the Doppler step, so the Doppler traces alone are not enough.')


def doppler_for(D):
    # keep every D-th sanitized CFR packet, then recompute Doppler with SHARP's settings (31 packets, sliding 1, -1.2)
    if D == 1:
        return DOPPLER_DIR
    doppler_dir = f'/kaggle/working/doppler_D{D}'
    if all(find_result(f'loeo_D{D}_{n}_fold{f}.pkl') for n in MODELS for f in FOLDS):
        return doppler_dir
    if not os.path.exists('/kaggle/working/SHARP'):
        subprocess.run(['git', 'clone', '-q', 'https://github.com/francescamen/SHARP', '/kaggle/working/SHARP'], check=True)
    os.makedirs(doppler_dir, exist_ok=True)
    for s in BEDROOM + ['S6a', 'S6b', 'S7a']:
        if os.path.exists(f'{doppler_dir}/{s}'):
            continue
        tmp = f'/kaggle/working/cfr_D{D}/{s}'
        os.makedirs(tmp, exist_ok=True)
        for f in glob.glob(f'{CFR_DIR}/{s}/*.mat'):
            cfr = sio.loadmat(f)['csi_matrix_processed']
            sio.savemat(f'{tmp}/{os.path.basename(f)}', {'csi_matrix_processed': cfr[::D]})
        subprocess.run([sys.executable, 'CSI_doppler_computation.py', f'/kaggle/working/cfr_D{D}/', s, f'{doppler_dir}/',
                        str(800 // D), str(800 // D), '31', '1', '-1.2'], cwd='/kaggle/working/SHARP/Python_code', check=True)
        shutil.rmtree(tmp)
    return doppler_dir
"""

EXP4_RUN = """
for D in D_LIST:
    doppler_dir = doppler_for(D)
    for fold in FOLDS:
        for name in MODELS:
            run_loeo(D, fold, name, doppler_dir)
"""

EXP4_REPORT = """
classes = CLASSES[5]
rows, confusions, windows = [], [], []
for fold in FOLDS:
    for D in D_LIST:
        for name, label, kw in VARIANTS:
            r = find_result(f'loeo_D{D}_{name}_fold{fold}.pkl')
            if r is None:
                continue
            e = evaluate(r['probs'], classes, logits=r['logits'], **kw)
            raw = evaluate(r['probs'], classes, logits=r['logits'], use_smoothing=False, **kw)
            rec = e['per_class_recall']
            row = {'Fold': fold, 'D': D, 'Packet rate': round(RATE / D, 1), 'Max speed (m/s)': round(WAVELENGTH * RATE / D / 4, 2),
                   'Airtime (%)': 100 / D, 'Model': label, 'Epoch used': r['epoch'],
                   'Empty + sitting': round(np.mean([rec['E'], rec['L']]), 2),
                   'Walking, running, jumping': round(np.mean([rec['W'], rec['R'], rec['J']]), 2),
                   **flat(e), **{f'{k} [no smoothing]': v for k, v in flat(raw).items()}}
            if not kw:
                for n in range(1, NUM_ANTENNAS + 1):
                    row[f'mean_recall {n} ant'] = round(antenna_subsets(r['probs'], classes, n, logits=r['logits'])['mean_recall'][0], 2)
            rows.append(row)
            confusions.append(confusion_frame(e['confusion'], classes, Fold=fold, D=D, Model=label))
            if name == 'Doppler-ResAT':
                y = np.concatenate([y for _, y in r['probs']])
                windows.append({'Fold': fold, 'D': D, 'Window': r['window'], 'Test stride': r['test_stride'],
                                **dict(zip(classes, np.bincount(y, minlength=5)))})

tables = {'packet_rate': pd.DataFrame(rows), 'confusion': pd.concat(confusions), 'test_windows': pd.DataFrame(windows)}
for key, df in tables.items():
    df.to_csv(f'{OUT}/exp4_{key}.csv', index=False)
print('4-antenna fusion with smoothing (no smoothing in the [no smoothing] columns); recall in %')
for df in tables.values():
    display(df)

# static and dynamic groups against packet rate, airtime on a second axis
results = tables['packet_rate']
fig, axes = plt.subplots(1, len(FOLDS), figsize=(6 * len(FOLDS), 4), squeeze=False)
for ax, fold in zip(axes[0], FOLDS):
    for label, style in [('Doppler-ResAT', '-'), ('SHARP re-run (probability fusion)', '--')]:
        r = results[(results['Fold'] == fold) & (results['Model'] == label)].sort_values('Packet rate')
        ax.plot(r['Packet rate'], r['Empty + sitting'], style + 'o', label=f'{label.split(" (")[0]}: empty + sitting')
        ax.plot(r['Packet rate'], r['Walking, running, jumping'], style + 's', label=f'{label.split(" (")[0]}: walk, run, jump')
    ax.set(xscale='log', xlabel='Packet rate (packets/s)', ylabel='Recall (%)', title=f'Fold {fold}')
    ax.set_xticks([RATE / D for D in D_LIST], [f'{RATE / D:g}' for D in D_LIST], minor=False)
    ax.minorticks_off()
    top = ax.secondary_xaxis('top', functions=(lambda r: 100 * r / RATE, lambda a: a * RATE / 100))
    top.set_xticks([100 / D for D in D_LIST], [f'{100 / D:g}' for D in D_LIST])
    top.minorticks_off()
    top.set_xlabel('Airtime (%)')
    ax.grid(alpha=0.3)
axes[0][0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/exp4_packet_rate.png', dpi=200)
fig.savefig(f'{OUT}/exp4_packet_rate.pdf')
plt.show()
"""

# ───────────────────────── Ablation (SHARP's protocol) ─────────────────────────

ABL_CONFIG = """
EXP = 1  # SHARP's protocol, as in 2_SHARP_protocol_comparison
SEEDS = [0]
NUM_CLASSES = 5
STRIDE_TEST = 30
S1 = ['S1a', 'S1b', 'S1c']
TEST_SETS = {'S1': S1, 'S2': ['S2a', 'S2b'], 'S3': ['S3a'], 'S4': ['S4a', 'S4b'], 'S5': ['S5a'],
             'S6': ['S6a', 'S6b'], 'S7': ['S7a']}
"""

ABL_MODELS = """
ABLATIONS = {'Doppler-ResAT-noCBAM': {'use_cbam': False}, 'Doppler-ResAT-noBiLSTM': {'use_bilstm': False}}
RECIPES.update({name: RECIPES['Doppler-ResAT'] for name in ABLATIONS})  # same training recipe as the full model
LABELS = {'Doppler-ResAT': 'Doppler-ResAT (full)', 'Doppler-ResAT-noCBAM': 'without CBAM (attend)',
          'Doppler-ResAT-noBiLSTM': 'without BiLSTM (sequence)', 'SHARP': 'SHARP re-run'}
"""

ABL_RUN = """
classes = CLASSES[NUM_CLASSES]
master, recs = load_sessions(DOPPLER_DIR, S1, classes)
for name in [*ABLATIONS, 'Doppler-ResAT']:  # the full model is reused from notebook 2's output when it is attached
    for seed in SEEDS:
        fname = f'exp{EXP}_{name}_seed{seed}.pkl'
        if find_result(fname):
            continue
        model, mean, std, epoch = train_model(name, master, recs, [(0, 0.6), (0.6, 0.8)], len(classes), seed)
        model.save_weights(f'{OUT}/exp{EXP}_{name}_seed{seed}.weights.h5')
        probs = predict_sets(model, DOPPLER_DIR, TEST_SETS, classes, {'S1': 0.8}, STRIDE_TEST, mean, std)
        save_result(fname, {'model': name, 'seed': seed, 'probs': probs, 'logits': RECIPES[name]['logits'],
                            'epoch': epoch, 'train_stride': RECIPES[name]['stride'], 'test_stride': STRIDE_TEST})
del master
"""

ABL_REPORT = """
main, per_class = [], []
for name, label in LABELS.items():
    r = find_result(f'exp{EXP}_{name}_seed{SEEDS[0]}.pkl')
    if r is None:
        continue
    row = {'Model': label, 'Epoch used': r['epoch']}
    for set_name in TEST_SETS:
        e = evaluate(r['probs'][set_name], classes, logits=r['logits'])
        row[set_name] = e['mean_recall']
        per_class.append({'Set': set_name, 'Model': label, **e['per_class_recall']})
    row['Mean S1-S7'] = round(float(np.mean([row[s] for s in TEST_SETS])), 2)
    main.append(row)

tables = {'main': pd.DataFrame(main), 'per_class': pd.DataFrame(per_class)}
for key, df in tables.items():
    df.to_csv(f'{OUT}/ablation_{key}.csv', index=False)
print(f'Mean recall (%), seed {SEEDS[0]}, 4-antenna fusion with smoothing')
display(tables['main'])
print('Per-class recall (%) on S4, S5, S7')
display(tables['per_class'][tables['per_class']['Set'].isin(['S4', 'S5', 'S7'])])
"""

ZIP = "!cd /kaggle/working && zip -qr results.zip results"

# ─────────────────────────────── notebooks ───────────────────────────────

def intro(title, body):
    return title + "\n\n" + body.strip() + """

Turn on a GPU. To continue in a new session, attach the previous `results` folder as a dataset: finished runs are skipped and included in the tables. Everything is written to `/kaggle/working/results` and zipped to `results.zip`."""


def shared():
    return [("code", SETUP), ("markdown", MODELS_MD), ("code", MODELS_CODE), ("markdown", DATA_MD),
            ("code", DATA_CODE), ("markdown", METRICS_MD), ("code", METRICS_CODE)]


NOTEBOOKS = {
    '1_LOEO_comparison.ipynb': [
        ("markdown", intro("# SHARP vs Doppler-ResAT on the same LOEO folds", """
Fold 1 holds out the bedroom and trains on S6a, S6b, S7a; Fold 2 holds out the living room and trains on the nine bedroom sessions plus S7a. Within the training sessions each activity stream is split in time (75% training, 25% validation) with windows cut separately on each side; 5 classes (E, L, W, R, J). Doppler-ResAT uses your trained fold models when they are attached (folders `fold_1/`, `fold_2/` with `model.keras` and `norm_stats.pkl`), otherwise it is retrained with the same settings. SHARP's classifier (`network_utility.py`) is trained with its own recipe on the same split, and both are tested on the same held-out windows (stride 17). Input: the `doppler_traces` dataset.

Doppler-ResAT on Fold 2 needs about 17 h on a T4, more than Kaggle's 12 h limit. Its training state is saved after every epoch and the run stops before the limit; run the notebook again with the previous output attached and training continues from the last saved epoch. The tables include every model finished so far.""")),
        ("code", LOEO_CONFIG), *shared(), ("code", LOEO_CODE), ("markdown", "## Train and test"), ("code", LOEO_RUN),
        ("markdown", "## Results"), ("code", LOEO_REPORT), ("code", ZIP)],
    '2_SHARP_protocol_comparison.ipynb': [
        ("markdown", intro("# SHARP vs Doppler-ResAT on SHARP's protocol: train on S1, test on S1–S7", """
Both models, each with its own training recipe, are trained on the same S1 time split (60% train, 20% validation) for each seed and tested on the same windows: the last 20% of S1, and S2–S7 at stride 30. SHARP's published numbers are shown alongside. 5 classes (E, L, W, R, J). Input: the `doppler_traces` dataset.""")),
        ("code", "EXP = 1" + EXP12_CONFIG[1]), *shared(), ("markdown", "## Train and test"), ("code", EXP12_RUN),
        ("markdown", "## Results"), ("code", SHARP_PAPER[1]), ("code", EXP12_REPORT), ("code", ZIP)],
    '3_ablation.ipynb': [
        ("markdown", intro("# Doppler-ResAT ablation on SHARP's protocol: without CBAM, without BiLSTM", """
Two Doppler-ResAT variants, each with one layer removed (CBAM, or the two BiLSTM layers, replaced by global average pooling), are trained with the full model's recipe on the same S1 time split (60% train, 20% validation) and tested on the same windows as notebook 2: the last 20% of S1, and S2–S7 at stride 30. Seed 0. Attach the output of `2_SHARP_protocol_comparison` so the full model and SHARP (seed 0) are reused instead of retrained. Input: the `doppler_traces` dataset.""")),
        ("code", ABL_CONFIG), *shared(), ("code", ABL_MODELS), ("markdown", "## Train and test"), ("code", ABL_RUN),
        ("markdown", "## Results"), ("code", ABL_REPORT), ("code", ZIP)],
}

for fname, cells in NOTEBOOKS.items():
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (t, src) in enumerate(cells):
        lines = src.strip("\n").split("\n")
        c = {"cell_type": t, "id": f"c{i}", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if t == "code":
            c.update(execution_count=None, outputs=[])
        nb["cells"].append(c)
    json.dump(nb, open(os.path.join(OUT_DIR, fname), "w"), indent=1)
