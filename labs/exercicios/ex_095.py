#!/usr/bin/env python3
"""EX-095 — Exemplo de erro

Aula 08 · B555 Redes Neurais
Dataset/imagem: erro_exemplo

Enunciado:
Mostre 1 caso em que o modelo erra (índice + features).

Rode (após clonar o repo):
  python labs/exercicios/ex_095.py
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
    titulo_ex("EX-095", "Exemplo de erro")
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    w = load_wine()
    X, y = w.data, (w.target == 0).astype(int)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, random_state=SEED, stratify=y)
    _, Xte, _, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=SEED, stratify=ytmp)
    m = Pipeline([("s", StandardScaler()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=SEED, early_stopping=True))])
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    err = np.where(pred != yte)[0]
    print("n_erros", len(err), "exemplo idx", err[0] if len(err) else None)
    fig, ax = plt.subplots()
    ax.bar(["acertos", "erros"], [(pred == yte).sum(), len(err)], color=["#0B6E4F", "#C1121F"])
    show_img_title(ax, "Erros no teste")
    plt.show()


if __name__ == "__main__":
    main()
