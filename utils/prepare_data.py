import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

def load_preprocessed_data(preprocessed_dir):
    point_clouds = []
    labels = []
    for file in os.listdir(preprocessed_dir):
        if file.endswith('_preprocessed.h5'):
            with h5py.File(os.path.join(preprocessed_dir, file), 'r') as f:
                clouds = f['point_clouds'][:]  # Load structured array
                lbls = f['labels'][:]          # Load labels
                # Extract fields into 3D array (n_samples, 1024, 4)
                n_samples = len(clouds)
                reshaped_clouds = np.zeros((n_samples, 1024, 4), dtype=np.float32)
                for i in range(n_samples):
                    reshaped_clouds[i, :, 0] = clouds[i]['x_cc']  # x_cc
                    reshaped_clouds[i, :, 1] = clouds[i]['y_cc']  # y_cc
                    reshaped_clouds[i, :, 2] = clouds[i]['vr_compensated']  # vr_compensated
                    reshaped_clouds[i, :, 3] = clouds[i]['rcs']    # rcs
                point_clouds.append(reshaped_clouds)
                labels.append(lbls)
    # Concatenate
    all_clouds = np.concatenate(point_clouds)
    all_labels = np.concatenate(labels)
    # Verify shape
    if all_clouds.shape[1] != 1024 or all_clouds.shape[2] != 4:
        raise ValueError(f"Unexpected shape {all_clouds.shape}. Expected (n, 1024, 4). Check preprocessing.")
    print(f"Loaded point clouds shape: {all_clouds.shape}, labels shape: {all_labels.shape}")
    return all_clouds, all_labels

def prepare_data():
    preprocessed_dir = 'data/preprocessed'
    output_dir = 'data/processed'

    # Load all data
    point_clouds, labels = load_preprocessed_data(preprocessed_dir)
    print(f"Total point clouds: {len(point_clouds)}, Total labels: {len(labels)}")

    # Split (80/20 train/val, stratify by label)
    train_clouds, val_clouds, train_labels, val_labels = train_test_split(
        point_clouds, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(os.path.join(output_dir, 'train_data.h5'), 'w') as f:
        f.create_dataset('point_clouds', data=train_clouds)
        f.create_dataset('labels', data=train_labels)
    with h5py.File(os.path.join(output_dir, 'val_data.h5'), 'w') as f:
        f.create_dataset('point_clouds', data=val_clouds)
        f.create_dataset('labels', data=val_labels)
    print(f"Train data saved: {len(train_clouds)} samples")
    print(f"Val data saved: {len(val_clouds)} samples")

if __name__ == '__main__':
    prepare_data()