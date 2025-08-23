import os
import numpy as np
import h5py
from pathlib import Path
from multiprocessing import Pool

def preprocess_sequence(args):
    seq_dir, output_dir, global_stats = args
    seq_name = Path(seq_dir).name
    h5_path = Path(seq_dir) / 'radar_data.h5'
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'radar_data' in f:
                radar = f['radar_data'][:]
                if 'label_id' in radar.dtype.names:
                    filtered = radar[(radar['label_id'] == 5) | (radar['label_id'] == 7)]
                else:
                    print(f"No 'label_id' field in {seq_name}. Skipping.")
                    return

                if 'track_id' in radar.dtype.names:
                    unique_tracks = np.unique(filtered['track_id'])
                    fused_point_clouds = []
                    labels = []
                    for track in unique_tracks:
                        track_data = filtered[filtered['track_id'] == track]
                        if len(track_data) < 5:
                            continue

                        original_cloud = np.zeros((1024, 4), dtype=np.float32)
                        n_points = min(len(track_data), 1024)
                        original_cloud[:n_points, 0] = track_data['x_cc'][:n_points]
                        original_cloud[:n_points, 1] = track_data['y_cc'][:n_points]
                        original_cloud[:n_points, 2] = track_data['vr_compensated'][:n_points]
                        original_cloud[:n_points, 3] = track_data['rcs'][:n_points]
                        if n_points < 1024:
                            original_cloud[n_points:] = np.nan
                        translated_cloud = original_cloud.copy()
                        translated_cloud[:, 0] += 0.5

                        # Fuse as (2, 1024, 4)
                        fused_cloud = np.stack([original_cloud, translated_cloud], axis=0)
                        fused_point_clouds.append(fused_cloud)

                        label = 5 if 5 in track_data['label_id'] else 7
                        labels.append(label)

                    if fused_point_clouds:
                        fused_point_clouds = np.array(fused_point_clouds)
                        labels = np.array(labels, dtype='int32')

                        # Normalize using global stats
                        for i, field in enumerate(['x_cc', 'y_cc', 'vr_compensated', 'rcs']):
                            valid = fused_point_clouds[:, :, :, i][~np.isnan(fused_point_clouds[:, :, :, i])]
                            if len(valid) > 0:
                                mean, std = global_stats[field]['mean'], global_stats[field]['std']
                                fused_point_clouds[:, :, :, i] = np.where(~np.isnan(fused_point_clouds[:, :, :, i]),
                                                                          (fused_point_clouds[:, :, :, i] - mean) / (std + 1e-6), np.nan)

                        # Save to HDF5
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f'{seq_name}_preprocessed.h5')
                        with h5py.File(output_path, 'w') as f_out:
                            f_out.create_dataset('fused_point_clouds', data=fused_point_clouds)
                            f_out.create_dataset('labels', data=labels)
                        print(f"Preprocessed {seq_name} saved to: {output_path}")
                    else:
                        print(f"No valid fused tracks found in {seq_name}.")
                else:
                    print(f"No 'track_id' field in {seq_name}. Skipping.")
            else:
                print(f"No 'radar_data' dataset in {seq_name}.")
    except Exception as e:
        print(f"Error processing {seq_name}: {e}")

def compute_global_stats(base_dir):
    all_x_cc, all_y_cc, all_vr, all_rcs = [], [], [], []
    for seq in range(1, 158):
        seq_dir = os.path.join(base_dir, f"sequence_{seq}")
        if os.path.exists(seq_dir):
            h5_path = Path(seq_dir) / 'radar_data.h5'
            with h5py.File(h5_path, 'r') as f:
                if 'radar_data' in f:
                    radar = f['radar_data'][:]
                    if 'label_id' in radar.dtype.names:
                        filtered = radar[(radar['label_id'] == 5) | (radar['label_id'] == 7)]
                        all_x_cc.extend(filtered['x_cc'])
                        all_y_cc.extend(filtered['y_cc'])
                        all_vr.extend(filtered['vr_compensated'])
                        all_rcs.extend(filtered['rcs'])
    stats = {}
    for field, data in [('x_cc', all_x_cc), ('y_cc', all_y_cc), ('vr_compensated', all_vr), ('rcs', all_rcs)]:
        data_array = np.array(data)
        valid = data_array[~np.isnan(data_array)]
        stats[field] = {'mean': np.mean(valid) if len(valid) > 0 else 0,
                        'std': np.std(valid) if len(valid) > 0 else 1}
    return stats

if __name__ == '__main__':
    base_dir = r"C:\Users\ameri\MmWave Radar\RadarScenes\data"
    output_dir = r"C:\Users\ameri\MmWave\data\preprocessed"

    print(f"Base directory: {base_dir}")
    print("Computing global normalization statistics...")
    global_stats = compute_global_stats(base_dir)
    print("Global Stats:", global_stats)

    seq_dirs = [os.path.join(base_dir, f"sequence_{seq}") for seq in range(1, 158) if os.path.exists(os.path.join(base_dir, f"sequence_{seq}"))]
    print(f"Found {len(seq_dirs)} sequence directories: {seq_dirs[:5]}...")
    args = [(seq_dir, output_dir, global_stats) for seq_dir in seq_dirs]

    with Pool(processes=4) as pool:
        pool.map(preprocess_sequence, args)

    print("Preprocessing completed for all sequences.")