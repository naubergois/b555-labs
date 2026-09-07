#!/usr/bin/env python3
"""EX-058 — MLP na janela

Aula 05 · B555 Redes Neurais
Dataset/imagem: mlp_serie

Enunciado:
Treine MLP e compare MAE com a baseline.

Rode (após clonar o repo):
  python labs/exercicios/ex_058.py
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
    titulo_ex("EX-058", "MLP na janela")
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
    from sklearn.neural_network import MLPRegressor
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    try:
        df = pd.read_csv(url)
        col = [c for c in df.columns if c.lower() != "month"][0]
        y = df[col].astype(float).values
    except Exception:
        t = np.arange(144)
        y = 100 + 0.5 * t + 20 * np.sin(t / 6)
    y = (y - y.mean()) / y.std()
    janela = 12
    X = np.array([y[i : i + janela] for i in range(len(y) - janela)])
    alvo = y[janela:]
    c = int(len(X) * 0.75)
    Xtr, Xte, ytr, yte = X[:c], X[c:], alvo[:c], alvo[c:]
    base = Xte[:, -1]
    m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=SEED, early_stopping=True)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    print("MAE MLP", mean_absolute_error(yte, pred), "vs base", mean_absolute_error(yte, base))
    fig, ax = plt.subplots()
    ax.plot(yte, label="real")
    ax.plot(pred, label="MLP")
    ax.plot(base, label="persistência", alpha=0.7)
    ax.legend()
    show_img_title(ax, "MLP vs persistência")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
