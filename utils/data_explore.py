import os
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from radar_scenes.labels import Label
from pathlib import Path


def explore_sequence(sequence_dir, output_dir='exploration'):
    scenes_path = Path(sequence_dir) / 'scenes.json'
    with open(scenes_path, 'r') as f:
        scenes_data = json.load(f)

    print("Sequence Name:", scenes_data["sequence_name"])
    print("Category:", scenes_data["category"])
    print("Number of Scenes:", len(scenes_data["scenes"]))
    print("First Timestamp:", scenes_data["first_timestamp"])
    print("Last Timestamp:", scenes_data["last_timestamp"])


    h5_path = Path(sequence_dir) / 'radar_data.h5'
    with h5py.File(h5_path, 'r') as f:
        # Explore odometry
        if 'odometry' in f:
            odometry = f['odometry']
            print("\nOdometry Shape:", odometry.shape)
            print("Odometry Fields:", odometry.dtype.names)
            odometry_data = np.array(odometry)
            print("Odometry Sample (first 5):", odometry_data[:5])

        # Explore radar_data
        if 'radar_data' in f:
            radar = f['radar_data']
            print("\nRadar Data Shape:", radar.shape)
            print("Radar Data Fields:", radar.dtype.names)
            radar_data = np.array(radar)
            print("Radar Data Sample (first 5):", radar_data[:5])

            # Unique labels
            if 'label_id' in radar.dtype.names:
                unique_labels = np.unique(radar_data['label_id'])
                print("\nUnique Label IDs:", unique_labels)
                try:
                    label_names = [Label(int(l)).name for l in unique_labels]
                    print("Label Names:", label_names)
                except NameError:
                    print("radar_scenes not installed; skipping label names.")
            else:
                print("No 'label_id' field.")

            # Summary statistics for relevant fields
            relevant_fields = ['x_cc', 'y_cc', 'vr_compensated', 'rcs', 'vr', 'range_sc', 'azimuth_sc']
            stats = {}
            for field in relevant_fields:
                if field in radar.dtype.names:
                    field_data = radar_data[field]
                    stats[field] = {
                        'min': float(np.min(field_data)),
                        'max': float(np.max(field_data)),
                        'mean': float(np.mean(field_data)),
                        'std': float(np.std(field_data))
                    }
            print("\nRelevant Fields Summary Statistics:", json.dumps(stats, indent=2))

            # Visualization: Scatter plot of x_cc vs y_cc colored by label_id (first 1000 points)
            if 'x_cc' in radar.dtype.names and 'y_cc' in radar.dtype.names and 'label_id' in radar.dtype.names:
                x = radar_data['x_cc'][:1000]
                y = radar_data['y_cc'][:1000]
                colors = radar_data['label_id'][:1000]
                plt.figure(figsize=(8, 6))
                plt.scatter(x, y, c=colors, cmap='viridis', s=10)
                plt.colorbar(label='Label ID')
                plt.xlabel('x_cc (m)')
                plt.ylabel('y_cc (m)')
                plt.title('Sample Radar Point Cloud (First 1000 Detections)')
                os.makedirs(output_dir, exist_ok=True)
                plot_path = os.path.join(output_dir, 'sample_point_cloud.png')
                plt.savefig(plot_path)
                plt.close()
                print("\nSample visualization saved to:", plot_path)
            else:
                print("\nCould not generate visualization (missing fields).")

            # Save sample pedestrian data (label_id == 7)
            if 'label_id' in radar.dtype.names:
                pedestrian_mask = radar_data['label_id'] == 7
                sample_pedestrian = radar_data[pedestrian_mask][:100]
                if len(sample_pedestrian) > 0:
                    sample_path = os.path.join(output_dir, 'sample_pedestrian.npy')
                    np.save(sample_path, sample_pedestrian)
                    print("\nSample pedestrian data saved to:", sample_path)
                else:
                    print("\nNo pedestrian data found.")
            else:
                print("\nCould not save sample (missing 'label_id').")
        else:
            print("\nNo 'radar_data' dataset found.")


if __name__ == '__main__':
    sequence_dir = r"C:\Users\ameri\MmWave Radar\RadarScenes\data\sequence_1"
    explore_sequence(sequence_dir)