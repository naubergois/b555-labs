#!/usr/bin/env python3
"""EX-024 — Validação vs teste

Aula 02 · B555 Redes Neurais
Dataset/imagem: val_vs_test

Enunciado:
Para escolher hiperparâmetros uso…

Rode (após clonar o repo):
  python labs/exercicios/ex_024.py
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
    titulo_ex("EX-024", "Validação vs teste")
    fig,ax=plt.subplots(figsize=(6,2.5)); ax.axis("off")
    for i,(t,c) in enumerate([("Treino","#A8DADC"),("Validação\n(escolhe)","#F4A261"),("Teste\n(uma vez)","#E76F51")]):
        ax.add_patch(plt.Rectangle((0.1+i*0.3,0.3),0.25,0.45,color=c,ec="k"))
        ax.text(0.225+i*0.3,0.52,t,ha="center",va="center",fontsize=9)
    show_img_title(ax,"Papéis do split"); plt.show()


if __name__ == "__main__":
    main()
