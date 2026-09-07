#!/usr/bin/env python3
"""EX-005 — Degrau vs sigmoid

Aula 01 · B555 Redes Neurais
Dataset/imagem: degrau_vs_sigmoid

Enunciado:
Troque a ativação por degrau. A fronteira fica suave ou abrupta?

Rode (após clonar o repo):
  python labs/exercicios/ex_005.py
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
    titulo_ex("EX-005", "Degrau vs sigmoid")
    z = np.linspace(-4,4,200)
    fig, ax = plt.subplots()
    ax.plot(z, 1/(1+np.exp(-z)), label="sigmoid")
    ax.plot(z, (z>=0).astype(float), label="degrau")
    ax.legend(); show_img_title(ax, "Degrau vs sigmoid")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
