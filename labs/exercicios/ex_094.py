#!/usr/bin/env python3
"""EX-094 — Manter baseline

Aula 08 · B555 Redes Neurais
Dataset/imagem: baseline_ok

Enunciado:
Manter o baseline é válido na rubrica?

Rode (após clonar o repo):
  python labs/exercicios/ex_094.py
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
    titulo_ex("EX-094", "Manter baseline")
    fig,ax=plt.subplots(figsize=(7,2)); ax.axis("off"); ax.text(0.5,0.5,'Manter o baseline é válido.',ha="center",fontsize=13); plt.show()


if __name__ == "__main__":
    main()
