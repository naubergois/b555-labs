#!/usr/bin/env python3
"""EX-011 — Escala das features

Aula 01 · B555 Redes Neurais
Dataset/imagem: scaler_box

Enunciado:
Por que StandardScaler ajuda antes do neurônio na Iris?

Rode (após clonar o repo):
  python labs/exercicios/ex_011.py
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
    titulo_ex("EX-011", "Escala das features")
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    iris=load_iris(); X=iris.data[:,2:4]
    fig, axes = plt.subplots(1,2, figsize=(8,3))
    axes[0].boxplot(X, labels=["petal_L","petal_W"]); axes[0].set_title("Antes")
    Xs=StandardScaler().fit_transform(X)
    axes[1].boxplot(Xs, labels=["petal_L","petal_W"]); axes[1].set_title("Depois StandardScaler")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
