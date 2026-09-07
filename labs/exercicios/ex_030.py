#!/usr/bin/env python3
"""EX-030 — Early na profunda

Aula 03 · B555 Redes Neurais
Dataset/imagem: early_epochs

Enunciado:
Early stopping reduz epochs? Quanto?

Rode (após clonar o repo):
  python labs/exercicios/ex_030.py
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
    titulo_ex("EX-030", "Early na profunda")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    bc=load_breast_cancer(); Xtr,_,ytr,_=train_test_split(bc.data,bc.target,test_size=0.3,random_state=SEED,stratify=bc.target)
    ns=[]
    for early in [False, True]:
        p=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,64,32),max_iter=600,early_stopping=early,validation_fraction=0.2,random_state=SEED))])
        p.fit(Xtr,ytr); ns.append(p.named_steps["m"].n_iter_); print("early",early,"epochs",ns[-1])
    fig,ax=plt.subplots(); ax.bar(["sem","com early"],ns,color="#0B6E4F"); show_img_title(ax,"Epochs")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
