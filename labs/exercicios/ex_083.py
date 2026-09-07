#!/usr/bin/env python3
"""EX-083 — Ablação tamanho

Aula 07 · B555 Redes Neurais
Dataset/imagem: ablacao_tam

Enunciado:
Compare MLP (16,) vs (64,32) na validação.

Rode (após clonar o repo):
  python labs/exercicios/ex_083.py
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
    titulo_ex("EX-083", "Ablação tamanho")
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    w = load_wine()
    X, y = w.data, (w.target == 0).astype(int)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    Xva, _, yva, _ = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    vals = []
    for h in [(16,), (64, 32)]:
        p = Pipeline([("s", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=h, max_iter=400, random_state=SEED, early_stopping=True))])
        p.fit(Xtr, ytr)
        vals.append(f1_score(yva, p.predict(Xva)))
        print(h, vals[-1])
    fig, ax = plt.subplots()
    ax.bar(["(16,)", "(64,32)"], vals)
    show_img_title(ax, "Ablação tamanho")
    plt.show()


if __name__ == "__main__":
    main()
