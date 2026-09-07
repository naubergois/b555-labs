#!/usr/bin/env python3
"""EX-071 — Teste uma vez

Aula 06 · B555 Redes Neurais
Dataset/imagem: teste_uma

Enunciado:
Avalie no teste só o escolhido. Registre os números.

Rode (após clonar o repo):
  python labs/exercicios/ex_071.py
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
    titulo_ex("EX-071", "Teste uma vez")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, roc_auc_score
    bc = load_breast_cancer()
    Xtr, Xtmp, ytr, ytmp = train_test_split(bc.data, bc.target, test_size=0.4, random_state=SEED, stratify=bc.target)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    mlp = Pipeline([("s", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=SEED, early_stopping=True))])
    mlp.fit(Xtr, ytr)
    print("TESTE F1", f1_score(yte, mlp.predict(Xte)), "AUC", roc_auc_score(yte, mlp.predict_proba(Xte)[:, 1]))


if __name__ == "__main__":
    main()
