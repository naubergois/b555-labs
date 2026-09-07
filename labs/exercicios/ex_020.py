#!/usr/bin/env python3
"""EX-020 — O que é loss

Aula 02 · B555 Redes Neurais
Dataset/imagem: loss_def

Enunciado:
Loss mede o quê em uma frase?

Rode (após clonar o repo):
  python labs/exercicios/ex_020.py
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
    titulo_ex("EX-020", "O que é loss")
    fig,ax=plt.subplots(figsize=(6,2)); ax.axis("off")
    ax.text(0.5,0.5,"Loss = medida do erro (previsão vs rótulo)", ha="center", fontsize=13)
    plt.show()


if __name__ == "__main__":
    main()
