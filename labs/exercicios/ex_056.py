#!/usr/bin/env python3
"""EX-056 — Janela 12

Aula 05 · B555 Redes Neurais
Dataset/imagem: janela_count

Enunciado:
Com janela=12, quantos pares (X,y) restam?

Rode (após clonar o repo):
  python labs/exercicios/ex_056.py
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
    titulo_ex("EX-056", "Janela 12")
    import pandas as pd
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    try:
        df = pd.read_csv(url)
        col = [c for c in df.columns if c.lower() != "month"][0]
        y = df[col].astype(float).values
    except Exception:
        t = np.arange(144)
        y = 100 + 0.5 * t + 20 * np.sin(t / 6)
    janela = 12
    n = len(y) - janela
    print("pares", n)
    fig, ax = plt.subplots()
    ax.bar(["n_pares"], [n])
    show_img_title(ax, f"janela={janela}")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
