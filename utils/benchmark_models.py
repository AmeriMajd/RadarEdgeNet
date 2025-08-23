import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import h5py
import time
from sklearn.metrics import accuracy_score, f1_score
import os


tf.keras.mixed_precision.set_global_policy('mixed_float16')



def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        clouds = f['fused_point_clouds'][:]  # Load as 4D array (n_samples, 2, 1024, 4)
        labels = f['labels'][:]
    reshaped_clouds = clouds.astype(np.float32)
    remapped_labels = np.where(labels == 5, 0, 1)
    return reshaped_clouds, remapped_labels

class NanMaskLayer(layers.Layer):
    def call(self, inputs):
        mask = ~tf.math.is_nan(tf.reduce_sum(inputs, axis=-1, keepdims=True))
        return tf.where(mask, inputs, tf.zeros_like(inputs))


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, clouds, labels, batch_size=32, translation_range=[-1, 1]):
        self.clouds = clouds
        self.labels = labels
        self.batch_size = batch_size
        self.translation_range = translation_range
        self.n = len(clouds)

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, idx):
        batch_clouds = self.clouds[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Apply random translation to second sub-cloud (x_cc only)
        for i in range(len(batch_clouds)):
            translation = np.random.uniform(self.translation_range[0], self.translation_range[1])
            batch_clouds[i, 1, :, 0] += translation  # Shift x_cc in second sub-cloud
        return batch_clouds, batch_labels



def mlp_model(num_views=2, num_points=1024, num_features=4):
    inputs = layers.Input(shape=(num_views, num_points, num_features))
    x = NanMaskLayer()(inputs)
    x = layers.Flatten()(x)  # Flatten to (2*1024*4 = 8192)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def pointnet_model(num_views=2, num_points=1024, num_features=4):
    inputs = layers.Input(shape=(num_views, num_points, num_features))
    view_outputs = []
    for i in range(num_views):
        x = inputs[:, i, :, :]
        x = NanMaskLayer()(x)
        x = layers.Conv1D(32, 1, activation='relu')(x)
        x = layers.Conv1D(64, 1, activation='relu')(x)
        x = layers.Conv1D(512, 1, activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=num_points)(x)
        view_outputs.append(x)
    x = layers.Concatenate()(view_outputs)  # Fuse views
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def pointnet_plus_model(num_views=2, num_points=1024, num_features=4):
    inputs = layers.Input(shape=(num_views, num_points, num_features))
    view_outputs = []
    for i in range(num_views):
        x = inputs[:, i, :, :]
        x = NanMaskLayer()(x)
        x = layers.Conv1D(32, 1, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 1, activation='relu')(x)
        view_outputs.append(x)
    x = layers.Concatenate()(view_outputs)  # Fuse views
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def dgcnn_model(num_views=2, num_points=1024, num_features=4):
    inputs = layers.Input(shape=(num_views, num_points, num_features))
    view_outputs = []
    for i in range(num_views):
        x = inputs[:, i, :, :]
        x = NanMaskLayer()(x)
        x = layers.Conv1D(32, 1, activation='relu')(x)
        x = layers.Conv1D(32, 1, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)
        view_outputs.append(x)
    x = layers.Concatenate()(view_outputs)  # Fuse views
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



def radarnet_model(num_views=2, num_points=1024, num_features=4):
    inputs = layers.Input(shape=(num_views, num_points, num_features))
    view_outputs = []
    for i in range(num_views):
        x = inputs[:, i, :, :]
        x = NanMaskLayer()(x)
        x = layers.Conv1D(32, 1, activation='relu')(x)
        x = layers.Conv1D(64, 1, activation='relu')(x)
        x = layers.Conv1D(32, 1, activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=64)(x)
        view_outputs.append(x)
    x = layers.Concatenate()(view_outputs)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model



def benchmark_model(model_func, name, train_clouds, train_labels, val_clouds, val_labels):
    model = model_func()
    train_gen = DataGenerator(train_clouds, train_labels)
    val_gen = DataGenerator(val_clouds, val_labels, translation_range=[0, 0])
    model.fit(train_gen, epochs=10, validation_data=val_gen, verbose=0)

    val_preds = model.predict(val_clouds, verbose=0)
    val_preds = np.argmax(val_preds, axis=1)
    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='weighted')

    start_time = time.time()
    for _ in range(100):
        model.predict(val_clouds[:1])
    inference_time = (time.time() - start_time) / 100 * 1000  # ms per frame

    size_mb = sum([tf.keras.backend.count_params(p) for p in model.trainable_weights]) * 4 / (1024 * 1024)
    return {'accuracy': acc, 'f1': f1, 'inference_ms': inference_time, 'size_mb': size_mb}



if __name__ == '__main__':
    train_clouds, train_labels = load_data('data/processed/train_data.h5')
    val_clouds, val_labels = load_data('data/processed/val_data.h5')

    models = {
        'MLP': mlp_model,
        'PointNet': pointnet_model,
        'PointNet++': pointnet_plus_model,
        'DGCNN': dgcnn_model,
        'RadarNet': radarnet_model
    }
    results = {}
    for name, model_func in models.items():
        print(f"Benchmarking {name}...")
        results[name] = benchmark_model(model_func, name, train_clouds, train_labels, val_clouds, val_labels)
    print("Results:", results)

    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_model = models[best_model_name]()
    train_gen = DataGenerator(train_clouds, train_labels)
    val_gen = DataGenerator(val_clouds, val_labels, translation_range=[0, 0])
    best_model.fit(train_gen, epochs=10, validation_data=val_gen, verbose=0)
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    os.makedirs('models', exist_ok=True)
    with open(f'models/{best_model_name}_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print(f"Best model ({best_model_name}) saved as .tflite")