#!/usr/bin/env python3
"""EX-055 — Série Airline

Aula 05 · B555 Redes Neurais
Dataset/imagem: airline_plot

Enunciado:
Plote a série Airline Passengers.

Rode (após clonar o repo):
  python labs/exercicios/ex_055.py
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
    titulo_ex("EX-055", "Série Airline")
    import pandas as pd
    url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    try:
      df=pd.read_csv(url); col=[c for c in df.columns if c.lower()!="month"][0]; y=df[col].astype(float).values
    except Exception:
      t=np.arange(144); y=100+0.5*t+20*np.sin(t/6)+RNG.normal(0,5,144)
    fig,ax=plt.subplots(); ax.plot(y,color="#1B4965"); show_img_title(ax,"Airline Passengers"); plt.tight_layout(); plt.show()
    globals()["_serie"]=y


if __name__ == "__main__":
    main()
