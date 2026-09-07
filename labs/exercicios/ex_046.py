#!/usr/bin/env python3
"""EX-046 — Convolução 3×3

Aula 04 · B555 Redes Neurais
Dataset/imagem: conv_manual

Enunciado:
Aplique média 3×3 num dígito e mostre antes/depois.

Rode (após clonar o repo):
  python labs/exercicios/ex_046.py
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
    titulo_ex("EX-046", "Convolução 3×3")
    from sklearn.datasets import load_digits
    img=load_digits().images[0]; k=np.ones((3,3))/9
    out=np.zeros((6,6))
    for i in range(6):
      for j in range(6):
        out[i,j]=np.sum(img[i:i+3,j:j+3]*k)
    fig,ax=plt.subplots(1,2,figsize=(6,3))
    ax[0].imshow(img,cmap="gray"); ax[0].set_title("antes"); ax[1].imshow(out,cmap="gray"); ax[1].set_title("média 3×3")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
