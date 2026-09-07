#!/usr/bin/env python3
"""EX-017 — Epochs efetivas

Aula 02 · B555 Redes Neurais
Dataset/imagem: epochs_bar

Enunciado:
Quantas epochs o Adam rodou (n_iter_)?

Rode (após clonar o repo):
  python labs/exercicios/ex_017.py
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
    titulo_ex("EX-017", "Epochs efetivas")
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    w=load_wine(); Xtr,_,ytr,_=train_test_split(w.data,w.target,test_size=0.25,random_state=SEED,stratify=w.target)
    pipe=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=400,random_state=SEED))])
    pipe.fit(Xtr,ytr); n=pipe.named_steps["m"].n_iter_
    fig,ax=plt.subplots(); ax.barh(["epochs"],[n], color="#0B6E4F"); show_img_title(ax,f"Epochs efetivas = {n}")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
