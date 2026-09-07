#!/usr/bin/env python3
"""EX-062 — Quando LSTM perde

Aula 05 · B555 Redes Neurais
Dataset/imagem: ruido_serie

Enunciado:
Aumente ruído sintético: LSTM ainda ganha?

Rode (após clonar o repo):
  python labs/exercicios/ex_062.py
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
    titulo_ex("EX-062", "Quando LSTM perde")
    t=np.arange(400); y=np.sin(0.05*t)+RNG.normal(0,0.5,400)
    fig,ax=plt.subplots(); ax.plot(y[:120]); show_img_title(ax,"Série com ruído alto"); plt.show()
    print("Com ruído alto, persistência costuma empatar ou ganhar da rede fraca.")


if __name__ == "__main__":
    main()
