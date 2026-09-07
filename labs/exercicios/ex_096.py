#!/usr/bin/env python3
"""EX-096 — Recomendação em 2 frases

Aula 08 · B555 Redes Neurais
Dataset/imagem: recomenda

Enunciado:
Escreva evidência → ação.

Rode (após clonar o repo):
  python labs/exercicios/ex_096.py
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
    titulo_ex("EX-096", "Recomendação em 2 frases")
    fig,ax=plt.subplots(figsize=(8,2.5)); ax.axis("off")
    ax.text(0.5,0.65,"Evidência: _______________________________",ha="center")
    ax.text(0.5,0.35,"Ação: ___________________________________",ha="center")
    show_img_title(ax,"Recomendação em 2 frases"); plt.show()


if __name__ == "__main__":
    main()
