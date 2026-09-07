#!/usr/bin/env python3
"""EX-048 — Sem Conv2D

Aula 04 · B555 Redes Neurais
Dataset/imagem: mlp_pixels

Enunciado:
Substitua por Dense-only e compare acc (1 epoch ok).

Rode (após clonar o repo):
  python labs/exercicios/ex_048.py
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
    titulo_ex("EX-048", "Sem Conv2D")
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    d=load_digits(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=0.25,random_state=SEED,stratify=d.target)
    sc=StandardScaler(); m=MLPClassifier(hidden_layer_sizes=(128,64),max_iter=200,random_state=SEED,early_stopping=True)
    m.fit(sc.fit_transform(Xtr),ytr); acc=m.score(sc.transform(Xte),yte); print("MLP pixels acc",acc)
    fig,ax=plt.subplots(); ax.bar(["MLP pixels"],[acc],color="#1B4965"); show_img_title(ax,"Baseline sem Conv"); plt.show()


if __name__ == "__main__":
    main()
