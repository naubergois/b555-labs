#!/usr/bin/env python3
"""EX-044 — Erros no teste

Aula 04 · B555 Redes Neurais
Dataset/imagem: cnn_erros

Enunciado:
Mostre 8 erros com y vs ŷ. Qual par mais comum?

Rode (após clonar o repo):
  python labs/exercicios/ex_044.py
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
    titulo_ex("EX-044", "Erros no teste")
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(SEED)
    (xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()
    xtr = xtr[:4000, ..., None].astype("float32") / 255.0
    ytr = ytr[:4000]
    xte = xte[:1000, ..., None].astype("float32") / 255.0
    yte = yte[:1000]
    names = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle"]
    m = keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(16, 3, activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(xtr, ytr, epochs=3, batch_size=64, validation_split=0.15, verbose=0)
    pred = m.predict(xte, verbose=0).argmax(1)
    err = np.where(pred != yte)[0][:8]
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for ax, i in zip(axes.ravel(), err):
        ax.imshow(xte[i].squeeze(), cmap="gray")
        ax.set_title(f"{names[yte[i]]}→{names[pred[i]]}", fontsize=7)
        ax.axis("off")
    plt.suptitle("Erros no teste"); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
