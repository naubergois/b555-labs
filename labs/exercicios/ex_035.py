#!/usr/bin/env python3
"""EX-035 — Noise 0,5

Aula 03 · B555 Redes Neurais
Dataset/imagem: moons_050

Enunciado:
Suba noise para 0,5 e repita o A/B (2 configs).

Rode (após clonar o repo):
  python labs/exercicios/ex_035.py
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
    titulo_ex("EX-035", "Noise 0,5")
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X,y=make_moons(n_samples=500,noise=0.5,random_state=SEED)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=SEED,stratify=y)
    fig,ax=plt.subplots(); ax.scatter(X[:,0],X[:,1],c=y,cmap="coolwarm",alpha=0.7)
    show_img_title(ax,"Moons noise=0.5"); plt.tight_layout(); plt.show()
    for nome,h,a in [("rasa",(8,),1e-4),("profunda",(64,64,32),0.0)]:
        p=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=h,alpha=a,max_iter=500,random_state=SEED))])
        p.fit(Xtr,ytr); print(nome, "te", f"{p.score(Xte,yte):.1%}", "gap", f"{p.score(Xtr,ytr)-p.score(Xte,yte):+.1%}")


if __name__ == "__main__":
    main()
