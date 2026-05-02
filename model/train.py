
"""
Treinamento do classificador de anomalias de vibração.
Versão: 1.0.0 | 2 classes | 64 features | FP32
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

def gerar_dataset(n=2000, n_features=64):
    """Simula leituras do acelerômetro: normal (0) e falha (1)."""
    X_normal = np.random.randn(n // 2, n_features) * 0.5
    X_falha  = np.random.randn(n // 2, n_features) * 1.8 + 0.3
    X = np.vstack([X_normal, X_falha]).astype(np.float32)
    y = np.array([0]*(n//2) + [1]*(n//2), dtype=np.int32)
    return X, y

def criar_modelo(n_features=64):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1,  activation='sigmoid')
    ], name='vibration_classifier_v1')

if __name__ == '__main__':
    X, y = gerar_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = criar_modelo()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_tr, y_tr, epochs=20, validation_split=0.1, verbose=1)
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f'Acuracia no teste: {acc:.4f}')
    model.save('model/artifacts/vibration_classifier.h5')
    print('Modelo salvo.')
