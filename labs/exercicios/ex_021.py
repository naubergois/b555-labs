#!/usr/bin/env python3
"""EX-021 — Batch vs epoch

Aula 02 · B555 Redes Neurais
Dataset/imagem: batch_epoch

Enunciado:
Uma epoch passa por quantos exemplos (em média)?

Rode (após clonar o repo):
  python labs/exercicios/ex_021.py
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
    titulo_ex("EX-021", "Batch vs epoch")
    fig,ax=plt.subplots(figsize=(7,2.5)); ax.axis("off")
    ax.text(0.1,0.6,"batch = pedaço · epoch = 1 passagem no treino", fontsize=12)
    ax.add_patch(plt.Rectangle((0.1,0.15),0.2,0.25,color="#A8DADC")); ax.add_patch(plt.Rectangle((0.35,0.15),0.2,0.25,color="#A8DADC"))
    ax.add_patch(plt.Rectangle((0.6,0.15),0.2,0.25,color="#457B9D")); ax.text(0.7,0.27,"epoch",ha="center")
    plt.show()


if __name__ == "__main__":
    main()
