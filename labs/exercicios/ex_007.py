#!/usr/bin/env python3
"""EX-007 — Nuvens próximas

Aula 01 · B555 Redes Neurais
Dataset/imagem: blobs_dist

Enunciado:
Aproxime as médias das blobs até a acurácia cair abaixo de 70%. Qual distância?

Rode (após clonar o repo):
  python labs/exercicios/ex_007.py
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
    titulo_ex("EX-007", "Nuvens próximas")
    from sklearn.datasets import make_blobs
    def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))
    for dist in [2.0, 1.0, 0.6, 0.35]:
        X,y = make_blobs(n_samples=160, centers=[[-dist,-dist],[dist,dist]], cluster_std=0.55, random_state=SEED)
        y=y.astype(float); w=RNG.normal(0,0.1,2); b=0.0
        for _ in range(50):
            p=sigmoid(X@w+b); e=y-p; w=w+0.8*(X.T@e)/len(X); b=b+0.8*float(e.mean())
        acc=((sigmoid(X@w+b)>=0.5).astype(float)==y).mean()
        print(f"dist={dist} acc={acc:.1%}")
        if dist in (2.0, 0.35):
            plot_fronteira(X,y,w,b,f"Blobs dist={dist}")


if __name__ == "__main__":
    main()
