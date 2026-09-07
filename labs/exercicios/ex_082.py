#!/usr/bin/env python3
"""EX-082 — Sprint baseline

Aula 07 · B555 Redes Neurais
Dataset/imagem: sprint_base

Enunciado:
Obtenha F1_val do baseline em <2 min de código.

Rode (após clonar o repo):
  python labs/exercicios/ex_082.py
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
    titulo_ex("EX-082", "Sprint baseline")
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    w = load_wine()
    X, y = w.data, (w.target == 0).astype(int)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    base = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=400, random_state=SEED))])
    base.fit(Xtr, ytr)
    print("F1_val", f1_score(yva, base.predict(Xva)))


if __name__ == "__main__":
    main()
