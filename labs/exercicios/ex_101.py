#!/usr/bin/env python3
"""EX-101 — Feedback útil

Aula 08 · B555 Redes Neurais
Dataset/imagem: feedback

Enunciado:
Feedback ácido sobre método é…

Rode (após clonar o repo):
  python labs/exercicios/ex_101.py
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
    titulo_ex("EX-101", "Feedback útil")
    fig,ax=plt.subplots(figsize=(7,2)); ax.axis("off"); ax.text(0.5,0.5,'Feedback de método = presente',ha="center",fontsize=13); plt.show()


if __name__ == "__main__":
    main()
