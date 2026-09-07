#!/usr/bin/env python3
"""EX-028 — A/B Breast Cancer

Aula 03 · B555 Redes Neurais
Dataset/imagem: ab_cancer

Enunciado:
Compare rasa vs profunda no mesmo split. Quem generaliza melhor?

Rode (após clonar o repo):
  python labs/exercicios/ex_028.py
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
    titulo_ex("EX-028", "A/B Breast Cancer")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    bc=load_breast_cancer(); Xtr,Xte,ytr,yte=train_test_split(bc.data,bc.target,test_size=0.3,random_state=SEED,stratify=bc.target)
    rows=[]
    for nome,h,a,e in [("rasa",(8,),1e-4,False),("profunda",(64,64,32),0.0,False)]:
        p=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=h,alpha=a,max_iter=500,early_stopping=e,random_state=SEED))])
        p.fit(Xtr,ytr); rows.append((nome,p.score(Xtr,ytr),p.score(Xte,yte)))
        print(nome, rows[-1][1:])
    fig,ax=plt.subplots(); ax.bar([r[0] for r in rows],[r[2] for r in rows], color="#1B4965")
    show_img_title(ax,"Acc teste — A/B Cancer"); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
