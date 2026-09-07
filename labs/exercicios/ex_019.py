#!/usr/bin/env python3
"""EX-019 — Early stopping

Aula 02 · B555 Redes Neurais
Dataset/imagem: early_stop_cmp

Enunciado:
Ligue early_stopping e compare epochs e gap.

Rode (após clonar o repo):
  python labs/exercicios/ex_019.py
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
    titulo_ex("EX-019", "Early stopping")
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X,y=make_classification(n_samples=220,n_features=20,n_informative=5,n_redundant=10,random_state=SEED)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=SEED,stratify=y)
    def fit(early):
        p=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(128,128,64),max_iter=800,early_stopping=early,validation_fraction=0.2,random_state=SEED))])
        p.fit(Xtr,ytr); return p.score(Xtr,ytr), p.score(Xte,yte), p.named_steps["m"].n_iter_
    a,b=fit(False),fit(True)
    fig,ax=plt.subplots(); ax.bar(["sem early\nte"],[a[1]], width=0.4, label="sem"); ax.bar([0.4], [b[1]], width=0.4, label="com")
    ax.legend(); show_img_title(ax,"Acc teste com/sem early"); plt.tight_layout(); plt.show()
    print("sem", a, "com", b)


if __name__ == "__main__":
    main()
