#!/usr/bin/env python3
"""Aula 04 — Visão: CNN simples (Keras se disponível) ou MLP em pixels (fallback).

Rode: python labs/04_cnn.py
No Colab: !pip install tensorflow  (já costuma vir instalado)
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42


def com_keras() -> bool:
    try:
        import tensorflow as tf  # noqa: F401

        return True
    except Exception:
        return False


def lab_keras() -> None:
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(SEED)
    (x_tr, y_tr), (x_te, y_te) = keras.datasets.fashion_mnist.load_data()
    x_tr = x_tr[:8000, ..., None].astype("float32") / 255.0
    y_tr = y_tr[:8000]
    x_te = x_te[:2000, ..., None].astype("float32") / 255.0
    y_te = y_te[:2000]

    modelo = keras.Sequential(
        [
            keras.layers.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(16, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = modelo.fit(
        x_tr,
        y_tr,
        epochs=5,
        batch_size=64,
        validation_split=0.15,
        verbose=2,
    )
    loss, acc = modelo.evaluate(x_te, y_te, verbose=0)
    print(f"\nFashion-MNIST teste: acc={acc:.2%} loss={loss:.4f}")
    print(f"val_acc última epoch: {hist.history['val_accuracy'][-1]:.2%}")
    print("Desafio: remova as Conv2D (só Dense em Flatten) e compare.")


def lab_sklearn_digits() -> None:
    print("TensorFlow ausente — fallback: dígitos 8×8 com MLP em pixels.\n")
    dig = load_digits()
    x = dig.data
    y = dig.target
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)

    # "CNN conceitual": MLP em pixels — baseline didático
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        random_state=SEED,
        early_stopping=True,
    )
    mlp.fit(x_tr_s, y_tr)
    y_hat = mlp.predict(x_te_s)
    print(f"Acurácia teste (MLP pixels): {mlp.score(x_te_s, y_te):.2%}")
    print(classification_report(y_te, y_hat, digits=3))

    # Mostra um "filtro" artesanal: média local 3×3 em uma imagem
    img = dig.images[0]
    kernel = np.ones((3, 3)) / 9.0
    # convolução válida manual
    h, w = img.shape
    out = np.zeros((h - 2, w - 2))
    for i in range(h - 2):
        for j in range(w - 2):
            out[i, j] = float(np.sum(img[i : i + 3, j : j + 3] * kernel))
    print(f"\nConvolução 3×3 (média) em digito[0]: {img.shape} → {out.shape}")
    print("Valores (arredondados):\n", np.round(out, 2))
    print("\nNo Colab, instale TensorFlow e rode de novo para ver Conv2D de verdade.")


def main() -> None:
    print("=== Lab 04 — Visão / CNN ===")
    if com_keras():
        lab_keras()
    else:
        lab_sklearn_digits()


if __name__ == "__main__":
    main()
