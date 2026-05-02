
"""
Quantizacao INT8 do modelo para TFLite Micro (ESP32).
Reduz FP32 para INT8: ~4x menor, ~2x mais rapido em hardware.
"""
import tensorflow as tf
import numpy as np

def quantizar_modelo(modelo_path, output_path, n_calib=200):
    """Converte Keras para TFLite INT8. Retorna tamanho em bytes."""
    converter = tf.lite.TFLiteConverter.from_keras_model(
        tf.keras.models.load_model(modelo_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def calib_dataset():
        for _ in range(n_calib):
            yield [np.random.randn(1, 64).astype(np.float32)]

    converter.representative_dataset = calib_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    return len(tflite_model)

if __name__ == '__main__':
    n = quantizar_modelo(
        'model/artifacts/vibration_classifier.h5',
        'model/artifacts/vibration_classifier_int8.tflite')
    print(f'Modelo INT8: {n:,} bytes ({n/1024:.1f} KB)')
