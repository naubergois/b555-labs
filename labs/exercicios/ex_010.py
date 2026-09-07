#!/usr/bin/env python3
"""EX-010 — Limiar 0,5

Aula 01 · B555 Redes Neurais
Dataset/imagem: limiar

Enunciado:
Com p=0,49 e p=0,51, quais classes no limiar 0,5?

Rode (após clonar o repo):
  python labs/exercicios/ex_010.py
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
    titulo_ex("EX-010", "Limiar 0,5")
    ps = [0.49, 0.51]
    fig, ax = plt.subplots()
    ax.bar(["p=0.49","p=0.51"], ps, color=["#3D5A80","#E07A5F"])
    ax.axhline(0.5, ls="--", color="k"); show_img_title(ax, "Limiar 0,5")
    plt.tight_layout(); plt.show()
    print("classes:", [int(p>=0.5) for p in ps])


if __name__ == "__main__":
    main()
