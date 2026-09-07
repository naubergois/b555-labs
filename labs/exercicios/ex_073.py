#!/usr/bin/env python3
"""EX-073 — Checklist relatório

Aula 06 · B555 Redes Neurais
Dataset/imagem: checklist_img

Enunciado:
Marque mentalmente os 7 itens do checklist.

Rode (após clonar o repo):
  python labs/exercicios/ex_073.py
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
    titulo_ex("EX-073", "Checklist relatório")
    itens=["Problema","Dados/ética","Split","Baseline","Métrica","Limites","Seed"]
    fig,ax=plt.subplots(figsize=(7,3)); ax.axis("off")
    for i,t in enumerate(itens):
        ax.text(0.05,0.9-i*0.12,f"☐ {t}",fontsize=12)
    show_img_title(ax,"Checklist do relatório"); plt.show()


if __name__ == "__main__":
    main()
