#!/usr/bin/env python3
"""EX-029 — L2 como freio

Aula 03 · B555 Redes Neurais
Dataset/imagem: l2_freio

Enunciado:
Com alpha=0,05 na profunda, o gap muda?

Rode (após clonar o repo):
  python labs/exercicios/ex_029.py
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
    titulo_ex("EX-029", "L2 como freio")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    bc=load_breast_cancer(); Xtr,Xte,ytr,yte=train_test_split(bc.data,bc.target,test_size=0.3,random_state=SEED,stratify=bc.target)
    gaps=[]
    for alpha in [0.0, 0.05]:
        p=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,64,32),alpha=alpha,max_iter=500,random_state=SEED))])
        p.fit(Xtr,ytr); gaps.append(p.score(Xtr,ytr)-p.score(Xte,yte)); print("alpha",alpha,"gap",gaps[-1])
    fig,ax=plt.subplots(); ax.bar(["sem L2","L2 0.05"],gaps,color="#E07A5F"); show_img_title(ax,"Gap treino−teste")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
