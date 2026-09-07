#!/usr/bin/env python3
"""EX-045 — Filtro compartilhado

Aula 04 · B555 Redes Neurais
Dataset/imagem: kernel_share

Enunciado:
Por que um kernel compartilhado economiza parâmetros?

Rode (após clonar o repo):
  python labs/exercicios/ex_045.py
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
    titulo_ex("EX-045", "Filtro compartilhado")
    fig,ax=plt.subplots(figsize=(7,2.5)); ax.axis("off")
    ax.text(0.5,0.55,"Mesmo filtro varre toda a imagem\n(= lanterna compartilhada)",ha="center",fontsize=12)
    show_img_title(ax,"Filtro compartilhado"); plt.show()


if __name__ == "__main__":
    main()
