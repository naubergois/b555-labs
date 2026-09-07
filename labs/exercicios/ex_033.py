#!/usr/bin/env python3
"""EX-033 — Escolha pela validação

Aula 03 · B555 Redes Neurais
Dataset/imagem: val_teste_frase

Enunciado:
Complete: escolho pela _____; guardo o _____ para o fim.

Rode (após clonar o repo):
  python labs/exercicios/ex_033.py
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
    titulo_ex("EX-033", "Escolha pela validação")
    fig,ax=plt.subplots(figsize=(7,2.2)); ax.axis("off")
    ax.text(0.5,0.5,'Escolho pela VALIDAÇÃO; guardo o TESTE para o fim.', ha="center", fontsize=12, wrap=True)
    show_img_title(ax, 'val_teste_frase'); plt.show()


if __name__ == "__main__":
    main()
