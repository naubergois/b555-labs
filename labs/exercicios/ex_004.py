#!/usr/bin/env python3
"""EX-004 — Efeito do bias

Aula 01 · B555 Redes Neurais
Dataset/imagem: bias_desloca

Enunciado:
Fixe w=[1,1] e compare b=−0,5 vs b=−1,5 num ponto x=[2,1]. A classe muda?

Rode (após clonar o repo):
  python labs/exercicios/ex_004.py
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
    titulo_ex("EX-004", "Efeito do bias")
    def sigmoid(z): return 1/(1+np.exp(-z))
    x = np.array([2.0, 1.0]); w = np.array([1.0, -1.0])
    for b in (-0.5, -1.5):
        p = float(sigmoid(x@w+b)); print(f"b={b}  p={p:.3f}  classe={int(p>=0.5)}")
    # imagem: deslocamento da fronteira 1D no logit
    zs = np.linspace(-3,3,100)
    fig, ax = plt.subplots()
    ax.plot(zs, 1/(1+np.exp(-(zs-0))), label="b=0")
    ax.plot(zs, 1/(1+np.exp(-(zs+1))), label="b deslocado")
    ax.axhline(0.5, ls="--", c="#888"); ax.legend(); show_img_title(ax, "Bias desloca a curva")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
