#!/usr/bin/env python3
"""Aula 05 — Sequências: janela deslizante + baseline vs MLP (LSTM no Colab).

Gera série senoidal ruidosa e prevê o próximo valor.
Rode: python labs/05_lstm_serie.py
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor

SEED = 42
RNG = np.random.default_rng(SEED)


def serie_sintetica(n: int = 800) -> np.ndarray:
    t = np.arange(n)
    return np.sin(0.05 * t) + 0.35 * np.sin(0.18 * t) + RNG.normal(0, 0.08, size=n)


def serie_airline() -> np.ndarray | None:
    """Airline Passengers (CSV público) — None se offline."""
    try:
        import csv
        import urllib.request
        from io import StringIO

        url = (
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
            "airline-passengers.csv"
        )
        raw = urllib.request.urlopen(url, timeout=20).read().decode()
        rows = list(csv.reader(StringIO(raw)))
        vals = [float(r[1]) for r in rows[1:] if len(r) > 1]
        return np.asarray(vals, dtype=float) if vals else None
    except Exception:
        return None


def janelas(y: np.ndarray, janela: int = 20):
    xs, ys = [], []
    for i in range(len(y) - janela):
        xs.append(y[i : i + janela])
        ys.append(y[i + janela])
    return np.asarray(xs), np.asarray(ys)


def baseline_persistencia(y_true: np.ndarray, y_prev: np.ndarray) -> dict:
    """Prevê o próximo valor = último valor conhecido."""
    return {
        "mae": mean_absolute_error(y_true, y_prev),
        "rmse": mean_squared_error(y_true, y_prev) ** 0.5,
    }


def main() -> None:
    y = serie_airline()
    if y is not None:
        print("Dataset livre: Airline Passengers (CSV público)", y.shape)
        janela = 12
    else:
        print("Offline — série sintética (mesmo protocolo)")
        y = serie_sintetica()
        janela = 24
    x, alvo = janelas(y, janela)
    n = len(x)
    corte = int(n * 0.75)
    x_tr, x_te = x[:corte], x[corte:]
    y_tr, y_te = alvo[:corte], alvo[corte:]

    # Baseline: persistência (último step da janela)
    y_base = x_te[:, -1]
    m_base = baseline_persistencia(y_te, y_base)
    print("=== Baseline persistência ===")
    print(f"  MAE={m_base['mae']:.4f}  RMSE={m_base['rmse']:.4f}")

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=400,
        random_state=SEED,
        early_stopping=True,
    )
    mlp.fit(x_tr, y_tr)
    y_hat = mlp.predict(x_te)
    mae = mean_absolute_error(y_te, y_hat)
    rmse = mean_squared_error(y_te, y_hat) ** 0.5
    print("\n=== MLP em janela (proxy de sequência) ===")
    print(f"  MAE={mae:.4f}  RMSE={rmse:.4f}")
    print(f"  ganho MAE vs baseline: {m_base['mae'] - mae:+.4f}")

    if _keras():
        print("\n=== LSTM Keras ===")
        _lstm(x_tr, y_tr, x_te, y_te)
    else:
        print("\nTensorFlow ausente — no Colab rode a célula LSTM do notebook.")

    print("\nDesafio: aumente o ruído e veja quando a MLP deixa de bater o baseline.")


def _keras() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def _lstm(x_tr, y_tr, x_te, y_te) -> None:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    tf.random.set_seed(SEED)
    x_tr3 = x_tr[..., None]
    x_te3 = x_te[..., None]
    modelo = keras.Sequential(
        [
            keras.layers.Input(shape=(x_tr3.shape[1], 1)),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    modelo.compile(optimizer="adam", loss="mse")
    modelo.fit(x_tr3, y_tr, epochs=8, batch_size=32, validation_split=0.15, verbose=2)
    pred = modelo.predict(x_te3, verbose=0).ravel()
    print(
        f"  LSTM MAE={mean_absolute_error(y_te, pred):.4f}  "
        f"RMSE={mean_squared_error(y_te, pred) ** 0.5:.4f}"
    )


if __name__ == "__main__":
    main()
