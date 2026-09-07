#!/usr/bin/env python3
"""EX-026 — Matriz de confusão

Aula 02 · B555 Redes Neurais
Dataset/imagem: confusion_wine

Enunciado:
Plote a matriz de confusão do Wine.

Rode (após clonar o repo):
  python labs/exercicios/ex_026.py
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
    titulo_ex("EX-026", "Matriz de confusão")
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    w=load_wine(); Xtr,Xte,ytr,yte=train_test_split(w.data,w.target,test_size=0.25,random_state=SEED,stratify=w.target)
    pipe=Pipeline([("s",StandardScaler()),("m",MLPClassifier(hidden_layer_sizes=(64,32),max_iter=400,random_state=SEED))])
    pipe.fit(Xtr,ytr); cm=confusion_matrix(yte, pipe.predict(Xte))
    fig,ax=plt.subplots(); im=ax.imshow(cm, cmap="Blues"); plt.colorbar(im,ax=ax)
    for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
        ax.text(j,i,cm[i,j],ha="center",va="center")
    show_img_title(ax,"Matriz de confusão — Wine"); ax.set_xlabel("ŷ"); ax.set_ylabel("y")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
