import h5py
import pandas as pd
import json
import os
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings
import joblib
import time

warnings.filterwarnings('ignore')

# Paths and config
DATA_ROOT = r"C:\Users\ameri\RadarScenes\data"
SELECTED_SENSOR_IDS = [1, 2]
MAX_TIME_DIFF = 0.05
MATCH_DIST_THRESHOLD = 2.0

# Confidence thresholds
PEDESTRIAN_CONFIDENCE = 0.7
CYCLIST_CONFIDENCE = 0.3
OTHER_CONFIDENCE = 0.8

# Reduced feature names
FEATURE_NAMES = [
    "x_cc", "y_cc", "vr_mean", "x_std", "y_std",
    "rcs_mean", "width", "num_points", "abs_vr_mean"
]


# --- Existing helpers ---
def predict_with_class_specific_confidence(model, X, thresholds):
    probs = model.predict_proba(X)
    predictions, confidence_scores = [], []
    for prob in probs:
        predicted_class = np.argmax(prob)
        max_prob = prob[predicted_class]
        if max_prob < thresholds[predicted_class]:
            predictions.append(2)  # default to "other"
        else:
            predictions.append(predicted_class)
        confidence_scores.append(max_prob)
    return np.array(predictions), np.array(confidence_scores)


def augment_minority_samples(X, y, target_samples=5000):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    X_augmented, y_augmented = [X], [y]
    for class_idx in [0, 1]:
        class_mask = (y == class_idx)
        X_class = X[class_mask]
        current_samples = class_counts[np.where(unique_classes == class_idx)[0][0]]
        if current_samples < target_samples:
            needed_samples = target_samples - current_samples
            multiplier = max(2, needed_samples // current_samples + 1)
            augmented_samples = []
            for sample in X_class:
                for _ in range(multiplier):
                    noise = np.random.normal(0, 0.05, sample.shape)
                    augmented_samples.append(sample + noise)
            augmented_samples = augmented_samples[:needed_samples]
            X_augmented.append(np.array(augmented_samples))
            y_augmented.append(np.ones(len(augmented_samples)) * class_idx)
    return np.vstack(X_augmented), np.concatenate(y_augmented)


def enhanced_summarize_snapshot(df):
    objects = defaultdict(dict)
    for track_id, group in df.groupby('track_id'):
        if group.empty or len(group) < 3:
            continue
        centroid = group[['x_cc', 'y_cc']].mean().values
        vr_mean = group['vr_compensated'].mean()
        x_std, y_std = group['x_cc'].std(), group['y_cc'].std()
        rcs_mean = group['rcs'].mean()
        width = group['x_cc'].max() - group['x_cc'].min()
        abs_vr_mean = abs(vr_mean)
        features = [centroid[0], centroid[1], vr_mean, x_std, y_std,
                    rcs_mean, width, len(group), abs_vr_mean]
        objects[track_id] = {"features": features, "label_id": group['label_id'].mode()[0]}
    return objects


def process_sequence(seq_name, group='train'):
    seq_path = os.path.join(DATA_ROOT, seq_name)
    h5_path, scenes_path = os.path.join(seq_path, 'radar_data.h5'), os.path.join(seq_path, 'scenes.json')
    if not os.path.exists(h5_path) or not os.path.exists(scenes_path):
        return []
    try:
        with open(scenes_path, 'r') as f:
            scenes_data = json.load(f)
    except:
        return []
    try:
        snapshots = []
        scenes_iter = scenes_data.get('scenes', [])
        if isinstance(scenes_iter, dict):
            scenes_iter = scenes_iter.values()
        for scene in scenes_iter:
            if 'odometry_timestamp' in scene:
                ts = scene['odometry_timestamp'] / 1e6
            elif 'timestamp' in scene:
                ts = scene['timestamp'] / 1e6
            elif 'prev_timestamp' in scene and 'next_timestamp' in scene:
                ts = (scene['prev_timestamp'] + scene['next_timestamp']) / 2e6
            else:
                continue
            sensor = scene['sensor_id']
            if sensor in SELECTED_SENSOR_IDS:
                snapshots.append({"timestamp": ts, "sensor_id": sensor, "start_idx": scene['radar_indices'][0],
                                  "end_idx": scene['radar_indices'][1]})
        snapshots = sorted(snapshots, key=lambda x: x['timestamp'])
    except:
        return []
    try:
        with h5py.File(h5_path, 'r') as h5:
            radar_data = h5['radar_data'][:]
    except:
        return []
    columns = ['timestamp', 'sensor_id', 'range_sc', 'azimuth_sc', 'rcs', 'vr', 'vr_compensated',
               'x_cc', 'y_cc', 'x_seq', 'y_seq', 'uuid', 'track_id', 'label_id']
    df_all = pd.DataFrame(radar_data, columns=columns)
    df_all['uuid'] = df_all['uuid'].str.decode('utf-8')
    df_all['track_id'] = df_all['track_id'].str.decode('utf-8')
    df_all['timestamp_sec'] = df_all['timestamp'] / 1e6
    df_all = df_all[(df_all['sensor_id'].isin(SELECTED_SENSOR_IDS)) & (df_all['track_id'] != '')]

    fused_samples, i = [], 0
    while i < len(snapshots) - 1:
        snap1 = snapshots[i]
        for j in range(i + 1, len(snapshots)):
            snap2 = snapshots[j]
            if snap1['sensor_id'] == snap2['sensor_id']:
                continue
            time_diff = abs(snap1['timestamp'] - snap2['timestamp'])
            if time_diff > MAX_TIME_DIFF:
                break
            df_snap1, df_snap2 = df_all.iloc[snap1['start_idx']:snap1['end_idx']], df_all.iloc[
                                                                                   snap2['start_idx']:snap2['end_idx']]
            objs1, objs2 = enhanced_summarize_snapshot(df_snap1), enhanced_summarize_snapshot(df_snap2)
            matched_pairs = {}
            for tid1, obj1 in objs1.items():
                best_match, min_dist = None, float('inf')
                for tid2, obj2 in objs2.items():
                    dist = np.linalg.norm(np.array(obj1['features'][:2]) - np.array(obj2['features'][:2]))
                    if dist < min_dist and dist < MATCH_DIST_THRESHOLD:
                        min_dist, best_match = dist, tid2
                if best_match:
                    matched_pairs[tid1] = best_match

            for tid1, tid2 in matched_pairs.items():
                obj1, obj2 = objs1[tid1], objs2[tid2]
                fused_features = [(f1 + f2) / 2 for f1, f2 in zip(obj1['features'], obj2['features'])]
                label_id = obj1['label_id'] if obj1['label_id'] == obj2['label_id'] else obj2['label_id']
                fused_samples.append((fused_features, label_id))
        i += 1
    return fused_samples


def samples_to_xy(samples):
    if not samples:
        return np.array([]), np.array([])
    X, y = np.array([feat for feat, _ in samples]), np.array([label for _, label in samples])

    def map_label(l): return 0 if l == 7 else (1 if l == 5 else 2)

    y = np.array([map_label(l) for l in y])
    return X, y


def save_model_to_pkl(model, output_path="radar_classifier.pkl"):
    joblib.dump(model, output_path, compress=3)
    size_bytes = os.path.getsize(output_path)
    print(f"Model saved → {output_path} | Size: {size_bytes / 1024 / 1024:.2f} MB")




# --- Main Execution ---
print("Loading sequence metadata...")
with open(os.path.join(DATA_ROOT, 'sequences.json'), 'r') as f:
    full_meta = json.load(f)
    sequences_meta = full_meta.get('sequences', {})

print(f"Total sequences: {len(sequences_meta)}")

all_samples, processed_count = [], 0
for seq_name, seq_info in sequences_meta.items():
    category = seq_info.get('category', 'unknown')
    print(f"Processing {seq_name} ({category})...")
    samples = process_sequence(seq_name, category)
    all_samples.extend(samples)
    processed_count += 1
    if processed_count % 10 == 0:
        print(f"Processed {processed_count}/{len(sequences_meta)} sequences → {len(all_samples)} samples")

print(f"\nTotal samples collected: {len(all_samples)}")
if not all_samples:
    print("No data collected. Exiting.")
    exit()

X, y = samples_to_xy(all_samples)
print(f"Raw samples shape: {X.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train samples: {X_train.shape}, Validation samples: {X_val.shape}")

scaler = StandardScaler()
X_train_scaled, X_val_scaled = scaler.fit_transform(X_train), scaler.transform(X_val)
print("Data scaled successfully")

print("\nAugmenting minority classes...")
X_train_aug, y_train_aug = augment_minority_samples(X_train_scaled, y_train, target_samples=5000)
print(f"After augmentation: {X_train_aug.shape}")

print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_aug, y_train_aug)
print(f"After SMOTE: {X_train_res.shape}")

print("\nTraining classifier...")
model = RandomForestClassifier(n_estimators=25, class_weight={0: 10, 1: 15, 2: 1}, random_state=42, max_depth=10,
                               min_samples_split=20)
model.fit(X_train_res, y_train_res)

# Print feature importances
print("\nFeature Importances:")
for name, importance in zip(FEATURE_NAMES, model.feature_importances_):
    print(f"{name}: {importance:.4f}")

save_model_to_pkl(model)

print("\n=== Evaluation ===")
thresholds = [PEDESTRIAN_CONFIDENCE, CYCLIST_CONFIDENCE, OTHER_CONFIDENCE]
n_runs, total_time = 10, 0
for _ in range(n_runs):
    start, _ = time.time(), None
    y_pred, _ = predict_with_class_specific_confidence(model, X_val_scaled, thresholds)
    total_time += (time.time() - start)
avg_time = total_time / n_runs
print(f"Avg inference time: {avg_time:.4f} sec")

print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=['pedestrian', 'cyclist', 'other']))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\n=== PROCESS COMPLETED ===")