#!/usr/bin/env python3
"""EX-081 — Wine binário setup

Aula 07 · B555 Redes Neurais
Dataset/imagem: wine_bin

Enunciado:
Transforme Wine em classe0 vs resto e mostre balance.

Rode (após clonar o repo):
  python labs/exercicios/ex_081.py
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
    titulo_ex("EX-081", "Wine binário setup")
    from sklearn.datasets import load_wine
    w=load_wine(); X,y=w.data,(w.target==0).astype(int)
    print("positivos", y.mean())
    globals().update({"X":X,"y":y})
    fig,ax=plt.subplots(); ax.bar(["0","1"],[ (y==0).sum(),(y==1).sum()]); show_img_title(ax,"Wine binário"); plt.show()


if __name__ == "__main__":
    main()
