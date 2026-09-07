#!/usr/bin/env python3
"""Aula 03 — Capacidade vs regularização (experimento A/B).

Compara MLP rasa × profunda × com L2/early_stopping no mesmo dataset.
Rode: python labs/03_regularizacao.py
"""
from __future__ import annotations

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def avaliar(nome: str, hidden, alpha: float, early: bool, x_tr, y_tr, x_te, y_te):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    max_iter=600,
                    random_state=SEED,
                    early_stopping=early,
                    validation_fraction=0.2,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )
    pipe.fit(x_tr, y_tr)
    tr = pipe.score(x_tr, y_tr)
    te = pipe.score(x_te, y_te)
    it = pipe.named_steps["mlp"].n_iter_
    print(f"{nome:28} treino={tr:.2%}  teste={te:.2%}  gap={tr-te:+.2%}  epochs={it}")
    return tr, te


def main() -> None:
    from sklearn.datasets import load_breast_cancer

    print("=== Dataset livre: Breast Cancer (sklearn/UCI) ===")
    bc = load_breast_cancer()
    x, y = bc.data, bc.target
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.3, random_state=SEED, stratify=y
    )

    print("=== A/B no mesmo split ===")
    configs = [
        ("A rasa (8,)", (8,), 1e-4, False),
        ("B profunda (64,64,32)", (64, 64, 32), 0.0, False),
        ("C profunda + L2", (64, 64, 32), 5e-2, False),
        ("D profunda + early", (64, 64, 32), 1e-3, True),
        ("E rasa + early", (8,), 1e-3, True),
    ]
    resultados = []
    for nome, hidden, alpha, early in configs:
        resultados.append(
            (nome, *avaliar(nome, hidden, alpha, early, x_tr, y_tr, x_te, y_te))
        )

    melhor = max(resultados, key=lambda t: t[2])
    print(f"\nMelhor no teste: {melhor[0]} ({melhor[2]:.2%})")
    print("\nDesafio: aumente o noise para 0.5 e repita. Quem sofre mais?")
    print("Escreva 5 linhas: capacidade sem regularização ≠ melhor modelo.")


if __name__ == "__main__":
    main()
