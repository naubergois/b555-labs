#!/usr/bin/env python3
"""EX-041 — Amostra Fashion-MNIST

Aula 04 · B555 Redes Neurais
Dataset/imagem: fmnist_grid

Enunciado:
Mostre um grid 2×5 de imagens do treino com rótulos.

Rode (após clonar o repo):
  python labs/exercicios/ex_041.py
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
    titulo_ex("EX-041", "Amostra Fashion-MNIST")
    try:
      from tensorflow import keras
      (x,y),_=keras.datasets.fashion_mnist.load_data()
      names=["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle"]
      fig,axes=plt.subplots(2,5,figsize=(8,3.5))
      for ax,i in zip(axes.ravel(), range(10)):
        ax.imshow(x[i],cmap="gray"); ax.set_title(names[y[i]],fontsize=8); ax.axis("off")
      plt.suptitle("Fashion-MNIST — amostras"); plt.tight_layout(); plt.show()
    except Exception as e:
      from sklearn.datasets import load_digits
      d=load_digits(); fig,axes=plt.subplots(2,5,figsize=(8,3.5))
      for ax,i in zip(axes.ravel(),range(10)):
        ax.imshow(d.images[i],cmap="gray"); ax.set_title(str(d.target[i])); ax.axis("off")
      plt.suptitle("Digits fallback"); plt.tight_layout(); plt.show(); print(e)


if __name__ == "__main__":
    main()
