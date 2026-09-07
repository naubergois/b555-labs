#!/usr/bin/env python3
"""EX-042 — CNN pequena

Aula 04 · B555 Redes Neurais
Dataset/imagem: cnn_train

Enunciado:
Treine 5 epochs a CNN didática. Qual val_acc final?

Rode (após clonar o repo):
  python labs/exercicios/ex_042.py
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
    titulo_ex("EX-042", "CNN pequena")
    try:
      import tensorflow as tf; from tensorflow import keras
      tf.random.set_seed(SEED)
      (xtr,ytr),(xte,yte)=keras.datasets.fashion_mnist.load_data()
      xtr=xtr[:6000,...,None].astype("float32")/255; ytr=ytr[:6000]
      xte=xte[:1500,...,None].astype("float32")/255; yte=yte[:1500]
      m=keras.Sequential([keras.layers.Input((28,28,1)),keras.layers.Conv2D(16,3,activation="relu"),keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32,3,activation="relu"),keras.layers.MaxPooling2D(),keras.layers.Flatten(),
        keras.layers.Dense(64,activation="relu"),keras.layers.Dropout(0.3),keras.layers.Dense(10,activation="softmax")])
      m.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
      h=m.fit(xtr,ytr,epochs=5,batch_size=64,validation_split=0.15,verbose=0)
      print("val_acc final", h.history["val_accuracy"][-1])
      globals()["_cnn_h"]=h; globals()["_cnn_m"]=m; globals()["_cnn_xte"]=xte; globals()["_cnn_yte"]=yte
    except Exception as e:
      print("TF indisponível", e)


if __name__ == "__main__":
    main()
