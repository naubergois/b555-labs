#!/usr/bin/env python3
"""Helpers compartilhados dos exercícios B555 (plots + seed)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

SEED = 42
RNG = np.random.default_rng(SEED)
plt.rcParams.update({"figure.figsize": (6, 3.5), "axes.grid": True, "grid.alpha": 0.3})


def titulo_ex(ex_id: str, titulo: str) -> None:
    try:
        from IPython.display import Markdown, display

        display(Markdown(f"### {ex_id} — {titulo}"))
    except Exception:
        print(f"\n=== {ex_id} — {titulo} ===")


def show_img_title(ax, title: str) -> None:
    ax.set_title(title, fontsize=11)


def plot_sigmoid():
    z = np.linspace(-6, 6, 200)
    s = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots()
    ax.plot(z, s, lw=2, color="#0B6E4F")
    ax.axhline(0.5, ls="--", color="#888")
    ax.axvline(0, ls="--", color="#888")
    ax.scatter([0], [0.5], s=60, zorder=3, color="#C45C26")
    show_img_title(ax, "Sigmoid σ(z) — imagem do exercício")
    ax.set_xlabel("z")
    ax.set_ylabel("σ(z)")
    plt.tight_layout()
    plt.show()


def plot_fronteira(X, y, w, b, title="Fronteira de decisão"):
    fig, ax = plt.subplots()
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#3D5A80", label="classe 0", alpha=0.7)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#E07A5F", label="classe 1", alpha=0.7)
    xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    if abs(w[1]) > 1e-6:
        ys = -(w[0] * xs + b) / w[1]
        ax.plot(xs, ys, "k-", lw=2, label="fronteira")
    show_img_title(ax, title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_loss(hist, title="Curva de loss"):
    fig, ax = plt.subplots()
    ax.plot(hist, color="#1B4965", lw=2)
    show_img_title(ax, title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    plt.tight_layout()
    plt.show()


def plot_diagrama_neuronio():
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.5, 2.5, "x₁"),
        (0.5, 1, "x₂"),
        (3.5, 1.75, "Σ w·x+b"),
        (6.2, 1.75, "φ(·)"),
        (8.5, 1.75, "ŷ"),
    ]
    for x, y, t in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), 1.4, 0.8, fill=True, color="#E8F1F2", ec="#1B4965", lw=2)
        )
        ax.text(x + 0.7, y + 0.4, t, ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(3.5, 2.1), xytext=(1.9, 2.9), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(3.5, 1.9), xytext=(1.9, 1.4), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(6.2, 2.15), xytext=(4.9, 2.15), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(8.5, 2.15), xytext=(7.6, 2.15), arrowprops=dict(arrowstyle="->"))
    show_img_title(ax, "Diagrama do neurônio — imagem do exercício")
    plt.tight_layout()
    plt.show()
