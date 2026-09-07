#!/usr/bin/env python3
"""EX-049 — Transfer learning (conceito)

Aula 04 · B555 Redes Neurais
Dataset/imagem: transfer_quiz

Enunciado:
Fine-tune usa pesos de onde?

Rode (após clonar o repo):
  python labs/exercicios/ex_049.py
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
    titulo_ex("EX-049", "Transfer learning (conceito)")
    fig,ax=plt.subplots(figsize=(7,2)); ax.axis("off"); ax.text(0.5,0.5,'Fine-tune: pesos de modelo pré-treinado',ha="center",fontsize=12); plt.show()


if __name__ == "__main__":
    main()
