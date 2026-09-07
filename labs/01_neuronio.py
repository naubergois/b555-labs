#!/usr/bin/env python3
"""Aula 01 — Neurônio artificial em NumPy puro.

Objetivo: ver peso, bias e ativação mudando a fronteira de decisão.
Rode: python labs/01_neuronio.py
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(42)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def neuronio(x: np.ndarray, w: np.ndarray, b: float, ativacao=sigmoid) -> np.ndarray:
    """y = ativacao(x · w + b). x shape (n, d), w shape (d,)."""
    return ativacao(x @ w + b)


def treinar_perceptron(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.5,
    epochs: int = 40,
) -> tuple[np.ndarray, float, list[float]]:
    """Regra delta simples (classificação binária 0/1)."""
    w = RNG.normal(0, 0.1, size=x.shape[1])
    b = 0.0
    historico: list[float] = []
    for _ in range(epochs):
        pred = neuronio(x, w, b)
        erro = y - pred
        # gradiente da BCE aproximado via (y - ŷ) * x
        w = w + lr * (x.T @ erro) / len(x)
        b = b + lr * float(erro.mean())
        loss = float(np.mean((y - pred) ** 2))
        historico.append(loss)
    return w, b, historico


def main() -> None:
    # Dataset livre: Iris (UCI via sklearn) — setosa vs versicolor, 2 features
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    mask = iris.target < 2
    x = StandardScaler().fit_transform(iris.data[mask][:, 2:4].astype(float))
    y = iris.target[mask].astype(float)
    print("Dataset livre: Iris (sklearn/UCI)", x.shape)

    print("=== Experimento 1: pesos manuais ===")
    for nome, w, b in [
        ("fronteira ruim", np.array([0.1, -0.8]), 0.0),
        ("fronteira ok", np.array([1.0, 1.0]), 0.0),
    ]:
        pred = (neuronio(x, w, b) >= 0.5).astype(float)
        acc = (pred == y).mean()
        print(f"  {nome}: acurácia={acc:.2%}  w={w}  b={b}")

    print("\n=== Experimento 2: treino automático ===")
    w, b, hist = treinar_perceptron(x, y, lr=0.8, epochs=50)
    pred = (neuronio(x, w, b) >= 0.5).astype(float)
    print(f"  w={w.round(3)}  b={b:.3f}")
    print(f"  loss inicial={hist[0]:.4f} → final={hist[-1]:.4f}")
    print(f"  acurácia treino={(pred == y).mean():.2%}")

    print("\n=== Desafio ===")
    print("1) Mude a ativação para degrau (lambda z: (z>=0).astype(float)) e compare.")
    print("2) Aproxime as duas nuvens (mude as médias) até o perceptron falhar.")
    print("3) Explique em 3 linhas: o que o bias desloca na fronteira?")


if __name__ == "__main__":
    main()
