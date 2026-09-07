#!/usr/bin/env python3
"""EX-059 — LSTM

Aula 05 · B555 Redes Neurais
Dataset/imagem: lstm_plot

Enunciado:
Treine LSTM e plote real vs pred vs persistência.

Rode (após clonar o repo):
  python labs/exercicios/ex_059.py
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
    titulo_ex("EX-059", "LSTM")
    import pandas as pd
    from sklearn.metrics import mean_absolute_error
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
    try:
        import tensorflow as tf
        from tensorflow import keras
        tf.random.set_seed(SEED)
        m = keras.Sequential([
            keras.layers.Input((janela, 1)),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])
        m.compile(optimizer="adam", loss="mse")
        m.fit(Xtr[..., None], ytr, epochs=25, batch_size=16, validation_split=0.15, verbose=0)
        pred = m.predict(Xte[..., None], verbose=0).ravel()
        print("MAE LSTM", mean_absolute_error(yte, pred))
    except Exception as e:
        from sklearn.neural_network import MLPRegressor
        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=SEED)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        print("LSTM fallback MLP", e, "MAE", mean_absolute_error(yte, pred))
    fig, ax = plt.subplots()
    ax.plot(yte, label="real")
    ax.plot(pred, label="modelo")
    ax.plot(base, label="persistência", alpha=0.7)
    ax.legend()
    show_img_title(ax, "Série — teste")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
