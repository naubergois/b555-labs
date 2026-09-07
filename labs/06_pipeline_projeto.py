#!/usr/bin/env python3
"""Aula 06 — Pipeline de projeto (template reutilizável).

Use como esqueleto do trabalho final: problema → split → baseline → modelo → métricas.
Rode: python labs/06_pipeline_projeto.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


@dataclass
class Relatorio:
    nome: str
    acc: float
    f1: float
    auc: float

    def linha(self) -> str:
        return f"{self.nome:22} acc={self.acc:.2%}  f1={self.f1:.3f}  auc={self.auc:.3f}"


def avaliar(nome: str, modelo, x_te, y_te) -> Relatorio:
    proba = modelo.predict_proba(x_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return Relatorio(
        nome=nome,
        acc=accuracy_score(y_te, pred),
        f1=f1_score(y_te, pred),
        auc=roc_auc_score(y_te, proba),
    )


def main() -> None:
    # Dataset livre padrão; no projeto final troque pelos SEUS dados
    from sklearn.datasets import load_breast_cancer

    bc = load_breast_cancer()
    x, y = bc.data, bc.target
    print("Dataset livre: Breast Cancer Wisconsin (sklearn/UCI)")
    # Alternativa sintética (descomente se quiser stress-test):
    # x, y = make_classification(
    #     n_samples=1200, n_features=16, n_informative=6, n_redundant=4,
    #     weights=[0.65, 0.35], random_state=SEED,
    # )
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        x, y, test_size=0.4, random_state=SEED, stratify=y
    )
    x_va, x_te, y_va, y_te = train_test_split(
        x_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp
    )
    print(f"shapes: tr={x_tr.shape} va={x_va.shape} te={x_te.shape}")
    print(f"positivos teste: {y_te.mean():.1%}")

    baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, random_state=SEED)),
        ]
    )
    baseline.fit(x_tr, y_tr)

    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=400,
                    random_state=SEED,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            ),
        ]
    )
    mlp.fit(x_tr, y_tr)

    print("\n=== Validação (escolha de modelo) ===")
    for r in [
        avaliar("baseline LR", baseline, x_va, y_va),
        avaliar("MLP", mlp, x_va, y_va),
    ]:
        print(" ", r.linha())

    print("\n=== Teste (só no final) ===")
    for r in [
        avaliar("baseline LR", baseline, x_te, y_te),
        avaliar("MLP", mlp, x_te, y_te),
    ]:
        print(" ", r.linha())

    print("\nChecklist do relatório:")
    for item in [
        "Problema de negócio em 5 linhas",
        "Origem dos dados + ética/privacidade",
        "Split e ausência de vazamento",
        "Baseline e por que a NN vale a pena (ou não)",
        "Métrica principal alinhada ao custo do erro",
        "Limitações e próximos passos",
        "Seed e instruções para reproduzir",
    ]:
        print(f"  [ ] {item}")


if __name__ == "__main__":
    main()
