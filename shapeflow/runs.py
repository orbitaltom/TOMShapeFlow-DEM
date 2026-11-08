"""Run naming and the run matrix built from config.json 'phases'."""

from __future__ import annotations


def fmt_bond(b: float) -> str:
    return "0" if b == 0 else f"{b:g}".replace(".", "p")


def hopper_run_id(phase, level, theta, outlet, gravity, bond, seed):
    return f"{phase}_{level}_hop{theta:g}_D{outlet:g}_{gravity}_Bo{fmt_bond(bond)}_s{seed}"


def chute_run_id(phase, level, angle, gravity, seed):
    return f"{phase}_{level}_chute{angle:g}_{gravity}_s{seed}"


def resolve_levels(cfg, spec):
    levels = list(cfg["levels"])
    if spec == "all":
        return levels
    if spec == "knee":
        knee = cfg["phases"].get("knee_level")
        if not knee:
            return None
        return list(dict.fromkeys([levels[0], knee, levels[-1]]))
    return list(spec)


def build_matrix(cfg, phases=None, knee=None):
    """List of run specs. Each spec: dict(kind, run_id, args for the run script)."""
    if knee:
        cfg = {**cfg, "phases": {**cfg["phases"], "knee_level": knee}}
    theta = cfg["hopper"]["wall_angle_from_vertical_deg"]
    specs, skipped = [], []
    for ph, p in cfg["phases"].items():
        if not isinstance(p, dict) or (phases and ph not in phases):
            continue
        levels = resolve_levels(cfg, p["levels"])
        if levels is None:
            skipped.append(ph)
            continue
        for level in levels:
            for seed in p["seeds"]:
                if p["kind"] == "chute":
                    for ang in p["angles"]:
                        specs.append({"phase": ph, "kind": "chute",
                                      "run_id": chute_run_id(ph, level, ang, p["gravity"], seed),
                                      "args": ["--level", level, "--angle", str(ang), "--gravity", p["gravity"],
                                               "--seed", str(seed), "--phase", ph]})
                else:
                    for outlet in p["outlets"]:
                        for bond in p["bond"]:
                            specs.append({"phase": ph, "kind": "hopper",
                                          "run_id": hopper_run_id(ph, level, theta, outlet, p["gravity"], bond, seed),
                                          "args": ["--level", level, "--outlet", str(outlet), "--gravity",
                                                   p["gravity"], "--bond", str(bond), "--seed", str(seed),
                                                   "--mode", p["mode"], "--phase", ph]})
    return specs, skipped
