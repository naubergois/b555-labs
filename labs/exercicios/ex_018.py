#!/usr/bin/env python3
"""EX-018 — Overfit proposital

Aula 02 · B555 Redes Neurais
Dataset/imagem: overfit_gap

Enunciado:
Rede grande + poucos dados: reporte gap treino−teste.

Rode (após clonar o repo):
  python labs/exercicios/ex_018.py
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
    titulo_ex("EX-018", "Overfit proposital")
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X,y=make_classification(n_samples=220,n_features=20,n_informative=5,n_redundant=10,random_state=SEED)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=SEED,stratify=y)
    p=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(128,128,64),max_iter=800,alpha=0.0,random_state=SEED))])
    p.fit(Xtr,ytr); tr,te=p.score(Xtr,ytr),p.score(Xte,yte)
    fig,ax=plt.subplots(); ax.bar(["treino","teste"],[tr,te], color=["#3D5A80","#E07A5F"])
    show_img_title(ax,f"Overfit gap={tr-te:+.1%}"); plt.tight_layout(); plt.show()
    print(tr, te, tr-te)


if __name__ == "__main__":
    main()
