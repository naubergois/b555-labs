#!/usr/bin/env python3
"""EX-006 — Geometria da fronteira

Aula 01 · B555 Redes Neurais
Dataset/imagem: fronteira_reta

Enunciado:
Com 2D, a equação w·x+b=0 é uma reta. O bias move o quê?

Rode (após clonar o repo):
  python labs/exercicios/ex_006.py
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
    titulo_ex("EX-006", "Geometria da fronteira")
    from sklearn.datasets import make_blobs
    X,y = make_blobs(n_samples=160, centers=[[-1,-1],[1.2,1]], cluster_std=0.5, random_state=SEED)
    y=y.astype(float); w=np.array([1.,1.]); b=0.
    plot_fronteira(X,y,w,b,"Fronteira = reta w·x+b=0")


if __name__ == "__main__":
    main()
