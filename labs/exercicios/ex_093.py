#!/usr/bin/env python3
"""EX-093 — Checklist oral

Aula 08 · B555 Redes Neurais
Dataset/imagem: checklist_oral

Enunciado:
Responda sim/não aos 5 itens do checklist oral.

Rode (após clonar o repo):
  python labs/exercicios/ex_093.py
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
    titulo_ex("EX-093", "Checklist oral")
    qs=["Explico o split?","Sei o baseline?","Recomendação em 2 frases?","Tenho 1 erro?","README roda?"]
    fig,ax=plt.subplots(figsize=(7,3.2)); ax.axis("off")
    for i,q in enumerate(qs):
        ax.text(0.05,0.9-i*0.16,f"☐ {q}",fontsize=12)
    show_img_title(ax,"Checklist oral"); plt.show()


if __name__ == "__main__":
    main()
