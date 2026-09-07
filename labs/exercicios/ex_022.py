#!/usr/bin/env python3
"""EX-022 — Gradiente

Aula 02 · B555 Redes Neurais
Dataset/imagem: gradiente_seta

Enunciado:
O gradiente aponta para onde a loss sobe ou desce?

Rode (após clonar o repo):
  python labs/exercicios/ex_022.py
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
    titulo_ex("EX-022", "Gradiente")
    fig,ax=plt.subplots(); xs=np.linspace(-2,2,100); ax.plot(xs, xs**2, lw=2)
    ax.annotate("gradiente sobe", xy=(1.2,1.44), xytext=(0.2,3), arrowprops=dict(arrowstyle="->", color="#C1121F"))
    ax.annotate("passo desce (−∇)", xy=(1.0,1.0), xytext=(1.5,2.5), arrowprops=dict(arrowstyle="->", color="#0B6E4F"))
    show_img_title(ax,"Descida de gradiente"); plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
