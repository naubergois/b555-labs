#!/usr/bin/env python3
"""EX-036 — Gap treino−teste

Aula 03 · B555 Redes Neurais
Dataset/imagem: gap_bars

Enunciado:
Plote barras do gap das 4 configs no Cancer.

Rode (após clonar o repo):
  python labs/exercicios/ex_036.py
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
    titulo_ex("EX-036", "Gap treino−teste")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    bc=load_breast_cancer(); Xtr,Xte,ytr,yte=train_test_split(bc.data,bc.target,test_size=0.3,random_state=SEED,stratify=bc.target)
    nomes,gaps=[],[]
    for nome,h,a,e in [("A",(8,),1e-4,False),("B",(64,64,32),0.,False),("C",(64,64,32),5e-2,False),("D",(64,64,32),1e-3,True)]:
        p=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=h,alpha=a,max_iter=500,early_stopping=e,validation_fraction=0.2,random_state=SEED))])
        p.fit(Xtr,ytr); nomes.append(nome); gaps.append(p.score(Xtr,ytr)-p.score(Xte,yte))
    fig,ax=plt.subplots(); ax.bar(nomes,gaps,color="#C45C26"); show_img_title(ax,"Gaps A/B/C/D"); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
