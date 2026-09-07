#!/usr/bin/env python3
"""EX-012 — O que o neurônio NÃO faz

Aula 01 · B555 Redes Neurais
Dataset/imagem: anti_antropo

Enunciado:
O neurônio 'entende' o significado de pétala? Sim ou não?

Rode (após clonar o repo):
  python labs/exercicios/ex_012.py
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
    titulo_ex("EX-012", "O que o neurônio NÃO faz")
    fig, ax = plt.subplots(figsize=(6,2.5)); ax.axis("off")
    ax.text(0.5,0.6,"Neurônio calcula σ(w·x+b).", ha="center", fontsize=13)
    ax.text(0.5,0.3,"Não 'entende' pétala nem cliente.", ha="center", fontsize=12, color="#C1121F")
    show_img_title(ax, "O que ainda não é"); plt.show()
    print("Resposta: NÃO")


if __name__ == "__main__":
    main()
