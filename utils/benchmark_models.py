import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import h5py
import time
from sklearn.metrics import accuracy_score, f1_score
import os

# Enable mixed precision for efficiency
tf.keras.mixed_precision.set_global_policy('mixed_float16')


# Load data function
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        clouds = f['point_clouds'][:]  # Load as 3D array (n_samples, 1024, 4)
        labels = f['labels'][:]
    n_samples = len(clouds)
    n_points = 1024
    n_features = 4
    reshaped_clouds = clouds  # Directly use the loaded 3D array
    remapped_labels = np.where(labels == 5, 0, 1)
    return reshaped_clouds, remapped_labels


# Custom layer to mask NaN values
class NanMaskLayer(layers.Layer):
    def call(self, inputs):
        mask = ~tf.math.is_nan(tf.reduce_sum(inputs, axis=-1, keepdims=True))
        return tf.where(mask, inputs, tf.zeros_like(inputs))


# MLP Model
def mlp_model(num_points=1024, num_features=4):
    inputs = layers.Input(shape=(num_points, num_features))
    x = NanMaskLayer()(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# PointNet Model
def pointnet_model(num_points=1024, num_features=4):
    inputs = layers.Input(shape=(num_points, num_features))
    x = NanMaskLayer()(inputs)
    x = layers.Conv1D(32, 1, activation='relu')(x)
    x = layers.Conv1D(64, 1, activation='relu')(x)
    x = layers.Conv1D(512, 1, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=num_points)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# PointNet++ Model
def pointnet_plus_model(num_points=1024, num_features=4):
    inputs = layers.Input(shape=(num_points, num_features))
    x = NanMaskLayer()(inputs)
    x = layers.Conv1D(32, 1, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 1, activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# DGCNN Model
def dgcnn_model(num_points=1024, num_features=4):
    inputs = layers.Input(shape=(num_points, num_features))
    x = NanMaskLayer()(inputs)
    x = layers.Conv1D(32, 1, activation='relu')(x)
    x = layers.Conv1D(32, 1, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Enhanced RadarNet Model
def radarnet_model(num_points=1024, num_features=4):
    inputs = layers.Input(shape=(num_points, num_features))
    x = NanMaskLayer()(inputs)
    x = layers.Conv1D(32, 1, activation='relu')(x)  # Increased from 16
    x = layers.Conv1D(64, 1, activation='relu')(x)  # Increased from 32
    x = layers.Conv1D(32, 1, activation='relu')(x)  # Added layer for depth
    x = layers.MaxPooling1D(pool_size=64)(x)  # Reduced pooling size for more features
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)  # Increased from 64
    x = layers.Dropout(0.2)(x)  # Added dropout to prevent overfitting
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Benchmark function
def benchmark_model(model_func, name, train_clouds, train_labels, val_clouds, val_labels):
    model = model_func()
    model.fit(train_clouds, train_labels, epochs=10, batch_size=32, validation_data=(val_clouds, val_labels),
              verbose=0)  # Increased epochs

    val_preds = model.predict(val_clouds, verbose=0)
    val_preds = np.argmax(val_preds, axis=1)
    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='weighted')

    start_time = time.time()
    for _ in range(100):
        model.predict(val_clouds[:1])  # Batch=1 for single-frame inference
    inference_time = (time.time() - start_time) / 100 * 1000  # ms per frame

    size_mb = sum([tf.keras.backend.count_params(p) for p in model.trainable_weights]) * 4 / (1024 * 1024)
    return {'accuracy': acc, 'f1': f1, 'inference_ms': inference_time, 'size_mb': size_mb}


# Main execution
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
    best_model.fit(train_clouds, train_labels, epochs=10, batch_size=32, validation_data=(val_clouds, val_labels),
                   verbose=0)
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
    tflite_model = converter.convert()
    os.makedirs('models', exist_ok=True)
    with open(f'models/{best_model_name}_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print(f"Best model ({best_model_name}) saved as .tflite")