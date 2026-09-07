#!/usr/bin/env python3
"""EX-072 — Distribuição de classes

Aula 06 · B555 Redes Neurais
Dataset/imagem: class_balance

Enunciado:
Plote % positivos em tr/va/te.

Rode (após clonar o repo):
  python labs/exercicios/ex_072.py
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
    titulo_ex("EX-072", "Distribuição de classes")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    bc = load_breast_cancer()
    Xtr, Xtmp, ytr, ytmp = train_test_split(bc.data, bc.target, test_size=0.4, random_state=SEED, stratify=bc.target)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    fig, ax = plt.subplots()
    ax.bar(["tr", "va", "te"], [ytr.mean(), yva.mean(), yte.mean()], color="#E07A5F")
    show_img_title(ax, "% positivos")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
