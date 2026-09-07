#!/usr/bin/env python3
"""EX-070 — MLP vs baseline

Aula 06 · B555 Redes Neurais
Dataset/imagem: escolha_val

Enunciado:
Compare F1/AUC na validação; escolha um modelo.

Rode (após clonar o repo):
  python labs/exercicios/ex_070.py
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
    titulo_ex("EX-070", "MLP vs baseline")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, roc_auc_score
    bc = load_breast_cancer()
    Xtr, Xtmp, ytr, ytmp = train_test_split(bc.data, bc.target, test_size=0.4, random_state=SEED, stratify=bc.target)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    base = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(max_iter=500, random_state=SEED))])
    mlp = Pipeline([("s", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=SEED, early_stopping=True))])
    base.fit(Xtr, ytr)
    mlp.fit(Xtr, ytr)
    rows = []
    for n, m in [("LR", base), ("MLP", mlp)]:
        f1 = f1_score(yva, m.predict(Xva))
        auc = roc_auc_score(yva, m.predict_proba(Xva)[:, 1])
        rows.append((n, f1, auc))
        print(n, f1, auc)
    fig, ax = plt.subplots()
    ax.bar([r[0] for r in rows], [r[1] for r in rows])
    show_img_title(ax, "F1 validação")
    plt.show()


if __name__ == "__main__":
    main()
