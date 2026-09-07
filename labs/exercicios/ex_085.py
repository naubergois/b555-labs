#!/usr/bin/env python3
"""EX-085 — Evidência|risco|ação

Aula 07 · B555 Redes Neurais
Dataset/imagem: era_template

Enunciado:
Preencha as 3 colunas em uma linha cada.

Rode (após clonar o repo):
  python labs/exercicios/ex_085.py
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
    titulo_ex("EX-085", "Evidência|risco|ação")
    fig,ax=plt.subplots(figsize=(8,3)); ax.axis("off")
    ax.text(0.02,0.7,"Evidência: ________________________________",fontsize=11)
    ax.text(0.02,0.45,"Risco: ____________________________________",fontsize=11)
    ax.text(0.02,0.2,"Ação: ____________________________________",fontsize=11)
    show_img_title(ax,"evidência | risco | ação"); plt.show()


if __name__ == "__main__":
    main()
