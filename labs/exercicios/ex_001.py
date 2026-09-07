#!/usr/bin/env python3
"""EX-001 — Pesos manuais na Iris

Aula 01 · B555 Redes Neurais
Dataset/imagem: fronteira_iris

Enunciado:
Com Iris binária (pétala), teste w=[0.2,-1] e w=[1,1]. Qual acurácia de cada um?

Rode (após clonar o repo):
  python labs/exercicios/ex_001.py
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
    titulo_ex("EX-001", "Pesos manuais na Iris")
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    iris = load_iris(); mask = iris.target < 2
    X = StandardScaler().fit_transform(iris.data[mask][:, 2:4].astype(float))
    y = iris.target[mask].astype(float)
    def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
    for nome,w,b in [("ruim", np.array([0.2,-1.0]), 0.0), ("ok", np.array([1.0,1.0]), 0.2)]:
        pred = (sigmoid(X@w+b)>=0.5).astype(float)
        print(nome, "acc=", f"{(pred==y).mean():.1%}")
        plot_fronteira(X, y, w, b, f"Iris — fronteira {nome}")


if __name__ == "__main__":
    main()
