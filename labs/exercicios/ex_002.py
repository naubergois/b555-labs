#!/usr/bin/env python3
"""EX-002 — Treino do neurônio

Aula 01 · B555 Redes Neurais
Dataset/imagem: loss_neuronio

Enunciado:
Treine 80 epochs e plote a loss. A loss final ficou abaixo de 0,05?

Rode (após clonar o repo):
  python labs/exercicios/ex_002.py
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
    titulo_ex("EX-002", "Treino do neurônio")
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    iris = load_iris(); mask = iris.target < 2
    X = StandardScaler().fit_transform(iris.data[mask][:, 2:4].astype(float))
    y = iris.target[mask].astype(float)
    def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
    w = RNG.normal(0,0.1,size=2); b=0.0; hist=[]
    for _ in range(80):
        pred = sigmoid(X@w+b); erro = y-pred
        w = w + 0.8*(X.T@erro)/len(X); b = b + 0.8*float(erro.mean())
        hist.append(float(np.mean((y-pred)**2)))
    plot_loss(hist, "Loss do neurônio na Iris")
    print("loss final=", hist[-1])


if __name__ == "__main__":
    main()
