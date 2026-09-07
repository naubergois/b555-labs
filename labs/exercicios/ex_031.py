#!/usr/bin/env python3
"""EX-031 — Capacidade

Aula 03 · B555 Redes Neurais
Dataset/imagem: capacidade_quiz

Enunciado:
Rede maior sempre é melhor? Sim/Não + por quê.

Rode (após clonar o repo):
  python labs/exercicios/ex_031.py
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
    titulo_ex("EX-031", "Capacidade")
    fig,ax=plt.subplots(figsize=(7,2.2)); ax.axis("off")
    ax.text(0.5,0.5,'Rede maior ≠ melhor. Regularize e valide.', ha="center", fontsize=12, wrap=True)
    show_img_title(ax, 'capacidade_quiz'); plt.show()


if __name__ == "__main__":
    main()
