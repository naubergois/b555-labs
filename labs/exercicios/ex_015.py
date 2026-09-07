#!/usr/bin/env python3
"""EX-015 — Wine + MLP

Aula 02 · B555 Redes Neurais
Dataset/imagem: wine_acc

Enunciado:
Treine MLP (64,32) no Wine. Qual acc no teste?

Rode (após clonar o repo):
  python labs/exercicios/ex_015.py
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
    titulo_ex("EX-015", "Wine + MLP")
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    w=load_wine(); Xtr,Xte,ytr,yte=train_test_split(w.data,w.target,test_size=0.25,random_state=SEED,stratify=w.target)
    pipe=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=400,random_state=SEED))])
    pipe.fit(Xtr,ytr); acc=accuracy_score(yte,pipe.predict(Xte)); print("acc teste=", f"{acc:.2%}")
    # imagem: barras por classe
    from collections import Counter
    c=Counter(yte); fig,ax=plt.subplots(); ax.bar(list(c.keys()), list(c.values()), color="#1B4965")
    show_img_title(ax,"Wine — classes no teste"); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
