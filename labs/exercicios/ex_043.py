#!/usr/bin/env python3
"""EX-043 — Curvas CNN

Aula 04 · B555 Redes Neurais
Dataset/imagem: cnn_curves

Enunciado:
Plote accuracy e loss de treino/val.

Rode (após clonar o repo):
  python labs/exercicios/ex_043.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_EXDIR = Path(__file__).resolve().parent
if str(_EXDIR) not in sys.path:
    sys.path.insert(0, str(_EXDIR))

from common import (  # noqa: E402
    RNG,
    SEED,
    np,
    plt,
    plot_diagrama_neuronio,
    plot_fronteira,
    plot_loss,
    plot_sigmoid,
    show_img_title,
    titulo_ex,
)


def main() -> None:
    titulo_ex("EX-043", "Curvas CNN")
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(SEED)
    (xtr, ytr), _ = keras.datasets.fashion_mnist.load_data()
    xtr = xtr[:4000, ..., None].astype("float32") / 255.0
    ytr = ytr[:4000]
    m = keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(16, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(xtr, ytr, epochs=3, batch_size=64, validation_split=0.15, verbose=0)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(h.history["loss"], label="tr"); ax[0].plot(h.history["val_loss"], label="val"); ax[0].legend(); ax[0].set_title("loss")
    ax[1].plot(h.history["accuracy"], label="tr"); ax[1].plot(h.history["val_accuracy"], label="val"); ax[1].legend(); ax[1].set_title("acc")
    plt.tight_layout(); plt.show()
    print("val_acc final", h.history["val_accuracy"][-1])


if __name__ == "__main__":
    main()
