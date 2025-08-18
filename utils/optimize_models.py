import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import h5py
import time
from sklearn.metrics import accuracy_score, f1_score
import os


# Load data function
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        clouds = f['point_clouds'][:]   # Load as 3D array (n_samples, 1024, 4)
        labels = f['labels'][:]
    reshaped_clouds = clouds.astype(np.float32)
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
    x = layers.Conv1D(64, 1, activation='relu')(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=num_points)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Quantize model function
def quantize_model(model, quantization_type, representative_data):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization_type == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization_type == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


# Benchmark TFLite model function
def benchmark_tflite_model(tflite_model, val_clouds, val_labels):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    val_preds = []
    start_time = time.time()
    for i in range(len(val_clouds)):
        input_data = val_clouds[i:i + 1]
        if input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        if output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

        val_preds.append(np.argmax(output_data))
    inference_time = (time.time() - start_time) / len(val_clouds) * 1000  # ms per sample

    acc = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='weighted')
    size_kb = len(tflite_model) / 1024
    return {'accuracy': acc, 'f1': f1, 'inference_ms': inference_time, 'size_kb': size_kb}


# Main execution
if __name__ == '__main__':
    train_clouds, train_labels = load_data('data/processed/train_data.h5')
    val_clouds, val_labels = load_data('data/processed/val_data.h5')

    def representative_data_gen():
        for i in range(100):
            yield [train_clouds[i:i + 1]]

    models = {
        'MLP': mlp_model,
        'PointNet': pointnet_model
    }

    results = {}
    for name, model_func in models.items():
        print(f"Training {name}...")
        model = model_func()
        model.fit(train_clouds, train_labels, epochs=10, batch_size=32,
                  validation_data=(val_clouds, val_labels), verbose=0)

        for q_type in ['dynamic', 'int8']:
            print(f"  Quantizing {name} with {q_type}...")
            tflite_model = quantize_model(model, q_type, representative_data_gen)
            results[f'{name}_{q_type}'] = benchmark_tflite_model(tflite_model, val_clouds, val_labels)

    # Print the results dictionary
    print("\nAll benchmark results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")
