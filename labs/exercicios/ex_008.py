#!/usr/bin/env python3
"""EX-008 — Derivada do peso

Aula 01 · B555 Redes Neurais
Dataset/imagem: sensibilidade

Enunciado:
∂z/∂xⱼ = wⱼ. Se w₁>0 e x₁ sobe, z sobe ou desce?

Rode (após clonar o repo):
  python labs/exercicios/ex_008.py
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
    titulo_ex("EX-008", "Derivada do peso")
    fig, ax = plt.subplots()
    xs = np.linspace(-2,2,50)
    ax.plot(xs, 1.5*xs, label="w=1.5 (sobe)")
    ax.plot(xs, -1.0*xs, label="w=-1 (desce)")
    ax.legend(); show_img_title(ax, "Sinal do peso = sensibilidade")
    plt.tight_layout(); plt.show()
    print("Se w1>0 e x1 sobe → z sobe")


if __name__ == "__main__":
    main()
