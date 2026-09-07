#!/usr/bin/env python3
"""EX-057 — Baseline persistência

Aula 05 · B555 Redes Neurais
Dataset/imagem: persistencia

Enunciado:
Calcule MAE/RMSE da persistência no teste.

Rode (após clonar o repo):
  python labs/exercicios/ex_057.py
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
    titulo_ex("EX-057", "Baseline persistência")
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error
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
    Xte, yte = X[c:], alvo[c:]
    base = Xte[:, -1]
    print("MAE", mean_absolute_error(yte, base), "RMSE", mean_squared_error(yte, base) ** 0.5)
    fig, ax = plt.subplots()
    ax.plot(yte, label="real")
    ax.plot(base, label="persistência", alpha=0.8)
    ax.legend()
    show_img_title(ax, "Baseline persistência")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
