#!/usr/bin/env python3
"""EX-009 — Regressão logística

Aula 01 · B555 Redes Neurais
Dataset/imagem: logreg_schema

Enunciado:
Um neurônio com sigmoid tem a mesma forma de qual baseline clássico?

Rode (após clonar o repo):
  python labs/exercicios/ex_009.py
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
    titulo_ex("EX-009", "Regressão logística")
    plot_diagrama_neuronio()
    print("Forma: p=σ(w·x+b) → regressão logística")


if __name__ == "__main__":
    main()
