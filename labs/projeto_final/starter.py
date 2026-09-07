#!/usr/bin/env python3
"""Starter do projeto final — troque o dataset e a métrica."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reusa o pipeline didático da aula 06 como ponto de partida
from importlib.util import module_from_spec, spec_from_file_location

spec = spec_from_file_location("pipeline06", ROOT / "06_pipeline_projeto.py")
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

if __name__ == "__main__":
    print("=== Starter projeto final B555 ===")
    print("Edite este arquivo: troque make_classification pelos SEUS dados.")
    mod.main()
