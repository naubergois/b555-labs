#!/usr/bin/env python3
"""EX-003 — Curva sigmoid

Aula 01 · B555 Redes Neurais
Dataset/imagem: sigmoid

Enunciado:
Plote σ(z) de −6 a 6. Em z=0, qual o valor?

Rode (após clonar o repo):
  python labs/exercicios/ex_003.py
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
    titulo_ex("EX-003", "Curva sigmoid")
    plot_sigmoid()
    print('σ(0) =', 1/(1+np.exp(0)))


if __name__ == "__main__":
    main()
