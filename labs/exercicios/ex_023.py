#!/usr/bin/env python3
"""EX-023 — Learning rate alto

Aula 02 · B555 Redes Neurais
Dataset/imagem: lr_esquema

Enunciado:
LR muito alto tipicamente faz a loss…

Rode (após clonar o repo):
  python labs/exercicios/ex_023.py
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
    titulo_ex("EX-023", "Learning rate alto")
    fig,ax=plt.subplots(); xs=np.linspace(0,10,50)
    ax.plot(xs, np.exp(-0.3*xs)+0.05*np.sin(3*xs), label="LR ok")
    ax.plot(xs, 0.5+0.4*np.sin(2*xs)+0.05*xs, label="LR alto (oscila)")
    ax.legend(); show_img_title(ax,"Learning rate"); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
