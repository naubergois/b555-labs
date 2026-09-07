#!/usr/bin/env python3
"""EX-065 — WhatsApp em ordem

Aula 05 · B555 Redes Neurais
Dataset/imagem: whatsapp

Enunciado:
A analogia de sequência desta trilha é…

Rode (após clonar o repo):
  python labs/exercicios/ex_065.py
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
    titulo_ex("EX-065", "WhatsApp em ordem")
    fig,ax=plt.subplots(figsize=(7,2)); ax.axis("off"); ax.text(0.5,0.5,'Analogia: WhatsApp em ordem',ha="center",fontsize=12); plt.show()


if __name__ == "__main__":
    main()
