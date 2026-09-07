#!/usr/bin/env python3
"""EX-092 — Roteiro 10 min

Aula 08 · B555 Redes Neurais
Dataset/imagem: roteiro_10

Enunciado:
Monte o outline 1–2–1–2–2–2 min.

Rode (após clonar o repo):
  python labs/exercicios/ex_092.py
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
    titulo_ex("EX-092", "Roteiro 10 min")
    fig,ax=plt.subplots(figsize=(8,3.5)); ax.axis("off")
    passos=["1' problema","2' dados/split","1' baseline","2' modelo","2' erros","2' recomendação"]
    for i,p in enumerate(passos):
        ax.add_patch(plt.Rectangle((0.05,0.85-i*0.13),0.9,0.11,color="#E8F1F2",ec="#1B4965"))
        ax.text(0.5,0.90-i*0.13,p,ha="center",va="center",fontsize=11)
    show_img_title(ax,"Roteiro 10 minutos"); plt.show()


if __name__ == "__main__":
    main()
