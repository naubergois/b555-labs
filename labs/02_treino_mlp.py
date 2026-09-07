#!/usr/bin/env python3
"""Aula 02 — Treino de MLP: loss, epochs e overfit.

Usa scikit-learn MLPClassifier com registro de loss_curve_.
Rode: python labs/02_treino_mlp.py
"""
from __future__ import annotations

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def treinar(x, y, hidden=(32, 16), max_iter=300, alpha=1e-4, early=False):
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
                    max_iter=max_iter,
                    random_state=SEED,
                    early_stopping=early,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                ),
            ),
        ]
    )
    pipe.fit(x, y)
    return pipe


def main() -> None:
    from sklearn.datasets import load_wine

    print("=== Dataset livre: Wine (sklearn/UCI) ===")
    wine = load_wine()
    x, y = wine.data, wine.target
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.25, random_state=SEED, stratify=y
    )

    modelo = treinar(x_tr, y_tr, hidden=(64, 32), max_iter=400, early=False)
    mlp = modelo.named_steps["mlp"]
    y_hat = modelo.predict(x_te)
    print(f"  epochs efetivas: {mlp.n_iter_}")
    print(f"  loss final: {mlp.loss_:.4f}")
    print(f"  acurácia teste: {accuracy_score(y_te, y_hat):.2%}")
    if getattr(mlp, "loss_curve_", None):
        curva = mlp.loss_curve_
        print(f"  loss[0]={curva[0]:.4f}  loss[-1]={curva[-1]:.4f}  pontos={len(curva)}")

    print("\n=== Overfit proposital (dados pouco ruidosos + rede grande) ===")
    x2, y2 = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=5,
        n_redundant=10,
        random_state=SEED,
    )
    x_tr2, x_te2, y_tr2, y_te2 = train_test_split(
        x2, y2, test_size=0.3, random_state=SEED, stratify=y2
    )
    gordo = treinar(x_tr2, y_tr2, hidden=(128, 128, 64), max_iter=800, alpha=0.0)
    print(f"  treino: {gordo.score(x_tr2, y_tr2):.2%}  teste: {gordo.score(x_te2, y_te2):.2%}")
    print("  (gap grande = overfit)")

    print("\n=== Com early_stopping ===")
    cedo = treinar(
        x_tr2, y_tr2, hidden=(128, 128, 64), max_iter=800, alpha=1e-3, early=True
    )
    print(f"  treino: {cedo.score(x_tr2, y_tr2):.2%}  teste: {cedo.score(x_te2, y_te2):.2%}")
    print(f"  epochs: {cedo.named_steps['mlp'].n_iter_}")

    print("\n=== Relatório (Wine) ===")
    print(classification_report(y_te, y_hat, digits=3))

    print("\nDesafio: plote mlp.loss_curve_ com matplotlib e marque early_stopping.")


if __name__ == "__main__":
    main()
