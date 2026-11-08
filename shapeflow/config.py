"""Load config.json and resolve package paths."""

from __future__ import annotations

import json
import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent

# Family numbers shared by every simulation script.
FAM_PARTICLE = 0      # active grains (DEME default clump family)
FAM_FROZEN = 1        # grains at rest in the catch bin/tray: fixed, no contact among themselves
FAM_GHOST = 2         # grains removed to break an arch: fixed, no contact with anything
FAM_WALL = 10         # 6061-T6 walls (fixed)
FAM_SLICE = 11        # frictionless slice walls of the hopper (fixed)
FAM_GATE = 12         # hopper gate while closed (fixed)
FAM_GATE_OPEN = 13    # hopper gate after opening (fixed, no contact with grains)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PKG_ROOT / "config.json"
    with open(path) as f:
        cfg = json.load(f)
    cfg["_path"] = str(path)
    return cfg


def gravity_value(cfg: dict, name: str) -> float:
    try:
        return float(cfg["gravity_m_s2"][name])
    except KeyError as exc:
        raise ValueError(f"gravity must be one of {list(cfg['gravity_m_s2'])}, got {name!r}") from exc


def templates_dir(cfg: dict) -> Path:
    return PKG_ROOT / cfg["shapes"]["templates_dir"]


def results_dir() -> Path:
    """results/ in the package, or the folder named by the SHAPEFLOW_RESULTS environment variable."""
    d = Path(os.environ["SHAPEFLOW_RESULTS"]) if os.environ.get("SHAPEFLOW_RESULTS") else PKG_ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cohesion_coefficient(bond: float, gravity: float) -> float:
    """DEME cohesion coefficient C (m/s^2) giving Bond number `bond` for two equal grains.

    The packaged model pulls with C * m_eff, and m_eff = m / 2 for equal grains, so
    Bo = (C * m / 2) / (m * g) = C / (2 g).
    """
    return 2.0 * bond * gravity
