#!/usr/bin/env python3
"""EX-091 — Revalidar protocolo

Aula 08 · B555 Redes Neurais
Dataset/imagem: defesa_numeros

Enunciado:
Rode baseline vs MLP e anote F1_val e F1_te.

Rode (após clonar o repo):
  python labs/exercicios/ex_091.py
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
    titulo_ex("EX-091", "Revalidar protocolo")
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    w=load_wine(); X,y=w.data,(w.target==0).astype(int)
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.4,random_state=SEED,stratify=y)
    Xva,Xte,yva,yte=train_test_split(Xtmp,ytmp,test_size=0.5,random_state=SEED,stratify=ytmp)
    base=Pipeline([("s",StandardScaler()),("lr",LogisticRegression(max_iter=400,random_state=SEED))])
    mlp=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=400,random_state=SEED,early_stopping=True))])
    base.fit(Xtr,ytr); mlp.fit(Xtr,ytr)
    rows=[("LR val",f1_score(yva,base.predict(Xva))),("MLP val",f1_score(yva,mlp.predict(Xva))),
          ("LR te",f1_score(yte,base.predict(Xte))),("MLP te",f1_score(yte,mlp.predict(Xte)))]
    print(rows)
    fig,ax=plt.subplots(); ax.bar([r[0] for r in rows],[r[1] for r in rows]); show_img_title(ax,"F1 defesa"); plt.xticks(rotation=20); plt.tight_layout(); plt.show()
    globals().update({"_mlp":mlp,"Xte":Xte,"yte":yte,"Xva":Xva})


if __name__ == "__main__":
    main()
