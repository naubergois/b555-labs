#!/usr/bin/env python3
"""EX-069 — Baseline LR

Aula 06 · B555 Redes Neurais
Dataset/imagem: baseline_auc

Enunciado:
Treine LogisticRegression. AUC na validação?

Rode (após clonar o repo):
  python labs/exercicios/ex_069.py
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
    titulo_ex("EX-069", "Baseline LR")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    bc = load_breast_cancer()
    Xtr, Xtmp, ytr, ytmp = train_test_split(bc.data, bc.target, test_size=0.4, random_state=SEED, stratify=bc.target)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    base = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=500, random_state=SEED))])
    base.fit(Xtr, ytr)
    auc = roc_auc_score(yva, base.predict_proba(Xva)[:, 1])
    print("AUC val", auc)
    fig, ax = plt.subplots()
    ax.bar(["AUC val"], [auc], color="#0B6E4F")
    plt.show()


if __name__ == "__main__":
    main()
