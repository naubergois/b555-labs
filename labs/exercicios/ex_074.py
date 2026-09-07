#!/usr/bin/env python3
"""EX-074 — Métrica e custo

Aula 06 · B555 Redes Neurais
Dataset/imagem: custo_metrica

Enunciado:
Se falso negativo é caro, que métrica priorizar?

Rode (após clonar o repo):
  python labs/exercicios/ex_074.py
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
    titulo_ex("EX-074", "Métrica e custo")
    fig,ax=plt.subplots(figsize=(7,2)); ax.axis("off"); ax.text(0.5,0.5,'Falso negativo caro → priorize recall',ha="center",fontsize=12); plt.show()


if __name__ == "__main__":
    main()
