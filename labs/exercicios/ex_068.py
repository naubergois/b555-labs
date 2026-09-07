#!/usr/bin/env python3
"""EX-068 — Shapes do split

Aula 06 · B555 Redes Neurais
Dataset/imagem: split_shapes

Enunciado:
Breast Cancer: imprima shapes tr/va/te (40/30/30 do resto).

Rode (após clonar o repo):
  python labs/exercicios/ex_068.py
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
    titulo_ex("EX-068", "Shapes do split")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    bc=load_breast_cancer(); X,y=bc.data,bc.target
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.4,random_state=SEED,stratify=y)
    Xva,Xte,yva,yte=train_test_split(Xtmp,ytmp,test_size=0.5,random_state=SEED,stratify=ytmp)
    print(Xtr.shape,Xva.shape,Xte.shape)
    globals().update({"Xtr":Xtr,"Xva":Xva,"Xte":Xte,"ytr":ytr,"yva":yva,"yte":yte})
    fig,ax=plt.subplots(); ax.bar(["tr","va","te"],[len(Xtr),len(Xva),len(Xte)],color="#1B4965")
    show_img_title(ax,"Shapes do split"); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
