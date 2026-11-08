"""Pieces shared by the chute, hopper and repose simulations (Chrono DEM-Engine, Python `deme`)."""

from __future__ import annotations

import csv
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (FAM_FROZEN, FAM_GATE, FAM_GATE_OPEN, FAM_GHOST, FAM_PARTICLE, FAM_SLICE, FAM_WALL,
                     templates_dir)


def import_deme():
    try:
        import deme  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Could not import 'deme'. Activate the DEME environment first "
                         "(see README: source ~/deme-env/bin/activate).") from exc
    return deme


# --------------------------------------------------------------------------
# Clump templates
# --------------------------------------------------------------------------

@dataclass
class ClumpTemplate:
    file: str
    mass: float
    moi: list
    radii: list
    positions: list
    bounding_radius: float
    n_spheres: int


def load_level(cfg: dict, level: str) -> list[ClumpTemplate]:
    tdir = templates_dir(cfg)
    with open(tdir / "templates_index.json") as f:
        index = json.load(f)
    if level not in index:
        raise ValueError(f"Unknown level {level!r}; available: {list(index)}")
    out = []
    for meta in index[level]:
        radii, pos = [], []
        with open(tdir / meta["file"]) as f:
            for row in csv.DictReader(f):
                pos.append([float(row["x"]), float(row["y"]), float(row["z"])])
                radii.append(float(row["r"]))
        out.append(ClumpTemplate(meta["file"], meta["mass_kg"], meta["moi_kg_m2"], radii, pos,
                                 meta["bounding_radius_m"], meta["n_spheres"]))
    return out


# --------------------------------------------------------------------------
# Solver, materials, families
# --------------------------------------------------------------------------

def make_solver(cfg: dict, device: int, cohesion_C: float | None = None, rough_floor: bool = False):
    """Create the solver and materials.

    Materials: 'particle', 'wall' (6061-T6), 'slice' (frictionless), and optionally 'rough'
    (a floor whose friction with grains equals grain-grain friction, used by the repose test).
    """
    deme = import_deme()
    s = deme.DEMSolver([device])
    s.SetVerbosity(cfg["solver"]["verbosity"])
    s.SetOutputFormat("CSV")
    s.SetOutputContent(["XYZ", "VEL", "FAMILY"])
    s.SetContactOutputFormat("CSV")
    s.SetContactOutputContent(["OWNER"])
    p, w = cfg["particles"], cfg["walls"]
    coh = cohesion_C is not None and cohesion_C > 0
    extra_p = {"Cohesion": float(cohesion_C)} if coh else {}
    extra_w = {"Cohesion": 0.0} if coh else {}
    mats = {
        "particle": s.LoadMaterial({"E": p["E_Pa"], "nu": p["nu"], "CoR": p["CoR"], "mu": p["mu"], "Crr": p["Crr"],
                                    **extra_p}),
        "wall": s.LoadMaterial({"E": w["E_Pa"], "nu": w["nu"], "CoR": w["CoR"], "mu": w["mu_particle_wall"],
                                "Crr": 0.0, **extra_w}),
        "slice": s.LoadMaterial({"E": w["E_Pa"], "nu": w["nu"], "CoR": w["CoR"], "mu": 0.0, "Crr": 0.0, **extra_w}),
    }
    if rough_floor:
        mats["rough"] = s.LoadMaterial({"E": w["E_Pa"], "nu": w["nu"], "CoR": w["CoR"], "mu": p["mu"], "Crr": 0.0,
                                        **extra_w})
    pm = mats["particle"]
    # Pairwise values are set explicitly so no default averaging rule decides them.
    s.SetMaterialPropertyPair("mu", pm, pm, p["mu"])
    s.SetMaterialPropertyPair("mu", pm, mats["wall"], w["mu_particle_wall"])
    s.SetMaterialPropertyPair("mu", pm, mats["slice"], w["mu_slice_walls"])
    if rough_floor:
        s.SetMaterialPropertyPair("mu", pm, mats["rough"], p["mu"])
    for m in mats.values():
        s.SetMaterialPropertyPair("CoR", pm, m, p["CoR"])
    # Rolling resistance: grain-grain and grain-6061 use particles.Crr; the frictionless slice walls get none.
    # The repose test's rough floor plane stands in for grains, so it gets the grain value.
    crr = float(p["Crr"])
    s.SetMaterialPropertyPair("Crr", pm, pm, crr)
    s.SetMaterialPropertyPair("Crr", pm, mats["wall"], crr)
    s.SetMaterialPropertyPair("Crr", pm, mats["slice"], 0.0)
    if rough_floor:
        s.SetMaterialPropertyPair("Crr", pm, mats["rough"], crr)
    if coh:
        s.SetMaterialPropertyPair("Cohesion", pm, pm, float(cohesion_C))
        for name, m in mats.items():
            if name != "particle":
                s.SetMaterialPropertyPair("Cohesion", pm, m, 0.0)
        # DEME's packaged cohesion model (Hertz-Mindlin plus C * m_eff pull while in contact).
        model_file = deme.GetDEMEDataFile("../kernel/DEMUserScripts/ForceModelWithCohesion.cu")
        force = s.ReadContactForceModel(model_file)
        force.SetPerContactWildcards({"delta_time", "delta_tan_x", "delta_tan_y", "delta_tan_z"})
        force.SetMustPairwiseMatProp({"CoR", "mu", "Crr", "Cohesion"})
    return s, mats


def load_clump_types(s, templates: list[ClumpTemplate], material):
    return [s.LoadClumpType(t.mass, t.moi, t.radii, t.positions, material) for t in templates]


def setup_families(s):
    """Fixed families and contact rules shared by every scene."""
    for fam in (FAM_FROZEN, FAM_GHOST, FAM_WALL, FAM_SLICE, FAM_GATE, FAM_GATE_OPEN):
        s.SetFamilyFixed(fam)
    # Frozen grains still hold up active grains, but skip contacts that cannot matter.
    for other in (FAM_FROZEN, FAM_WALL, FAM_SLICE, FAM_GATE, FAM_GATE_OPEN):
        s.DisableContactBetweenFamilies(FAM_FROZEN, other)
    # Ghosts (grains removed to break an arch) touch nothing.
    for other in (FAM_PARTICLE, FAM_FROZEN, FAM_GHOST, FAM_WALL, FAM_SLICE, FAM_GATE, FAM_GATE_OPEN):
        s.DisableContactBetweenFamilies(FAM_GHOST, other)
    # An opened gate no longer touches grains.
    s.DisableContactBetweenFamilies(FAM_GATE_OPEN, FAM_PARTICLE)
    # Fixed parts never need contact checks among themselves.
    fixed_parts = (FAM_WALL, FAM_SLICE, FAM_GATE, FAM_GATE_OPEN)
    for i, a in enumerate(fixed_parts):
        for b in fixed_parts[i:]:
            s.DisableContactBetweenFamilies(a, b)


def freeze_rule(s, condition_xyz: str, speed: float, gravity: float):
    """Freeze grains that satisfy `condition_xyz`, move slower than `speed` (m/s) and are supported.

    DEME's stored acceleration excludes gravity, so a resting grain shows roughly g of contact
    acceleration while a grain at the top of a bounce shows almost none; requiring at least g/2
    keeps grains from freezing in mid-air.
    """
    a2 = (0.5 * gravity) ** 2
    cond = (f"return ({condition_xyz}) && (vX * vX + vY * vY + vZ * vZ < {speed * speed:.6e})"
            f" && (accX * accX + accY * accY + accZ * accZ > {a2:.6e});")
    s.ChangeFamilyWhen(FAM_PARTICLE, FAM_FROZEN, cond)


def contact_multiplicity(s, path):
    """Mean number of touching sphere pairs per touching grain pair (clump-clump), from a contact file."""
    s.WriteContactFile(str(path))
    s.WaitForPendingOutput()
    pairs, rows = set(), 0
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("contact_type", "").strip() != "SS":
                continue
            a, b = int(row["A"]), int(row["B"])
            pairs.add((min(a, b), max(a, b)))
            rows += 1
    return (rows / len(pairs)) if pairs else float("nan")


def lookup_multiplicity(cfg, level, results):
    """Sphere pairs per grain contact for a level: config value, else the mean over finished runs."""
    v = cfg.get("cohesion", {}).get("sphere_pairs_per_contact", {}).get(level)
    if v:
        return float(v)
    vals = []
    for f in Path(results).glob("*/summary.json"):
        d = json.load(open(f))
        if d["meta"].get("level") == level:
            x = d["summary"].get("sphere_pairs_per_contact")
            if x and np.isfinite(x):
                vals.append(float(x))
    if not vals:
        raise SystemExit(f"No contact multiplicity for {level} yet. Run a non-cohesive hopper run of {level} "
                         f"first (Phase A), or set cohesion.sphere_pairs_per_contact.{level} in config.json.")
    return float(np.mean(vals))


def particle_states(s):
    """Positions, velocities and families of every owner (walls included; filter by family)."""
    n = int(s.GetNumOwners())
    pos = np.asarray(s.GetOwnerPosition(0, n), dtype=float).reshape(-1, 3)
    vel = np.asarray(s.GetOwnerVelocity(0, n), dtype=float).reshape(-1, 3)
    fam = np.asarray(s.GetOwnerFamily(0, n), dtype=int).reshape(-1)
    return pos, vel, fam


def common_solver_settings(s, cfg, gravity_vec):
    s.SetTimeStepSize(cfg["solver"]["time_step_s"])
    s.SetGravitationalAcceleration([float(g) for g in gravity_vec])
    s.SetMaxVelocity(cfg["solver"]["max_velocity_m_s"])
    s.SetExpandSafetyType("auto")


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------

def grid_points(x_range, y_range, z_range, spacing, rng=None, jitter=0.05):
    """Grid of centres inside the given ranges, `spacing` apart, with a small random jitter."""
    xs = np.arange(x_range[0], x_range[1] + 1e-12, spacing)
    ys = np.arange(y_range[0], y_range[1] + 1e-12, spacing)
    zs = np.arange(z_range[0], z_range[1] + 1e-12, spacing)
    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        return np.zeros((0, 3))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    if rng is not None and jitter > 0:
        P += rng.uniform(-jitter, jitter, P.shape) * spacing
    return P


def stage_positions(layer_fn, z_start, n_max, stage_height, spacing, rng, jitter=0.04):
    """Loose grid of up to n_max centres in layers from z_start up to z_start + stage_height.

    layer_fn(z) returns the centres of one horizontal layer at height z (already inside the walls).
    Filling in stages keeps drop heights short: a single loose column of elongated clumps can
    be metres tall.
    """
    pts, z = [], z_start
    while len(pts) < n_max and z <= z_start + stage_height:
        layer = layer_fn(z)
        if len(layer):
            pts.extend(layer.tolist())
        z += spacing
    P = np.asarray(pts[:n_max], dtype=float).reshape(-1, 3)
    if len(P):
        P += rng.uniform(-jitter, jitter, P.shape) * spacing
    return P


def settle(s, ke_insp, total_mass, sync, min_time, max_time, rms_target):
    """Advance until the rms grain speed drops below rms_target (after min_time) or max_time passes."""
    t, v = 0.0, float("inf")
    while t < max_time - 1e-12:
        s.DoDynamicsThenSync(sync)
        t += sync
        v = math.sqrt(max(2.0 * float(ke_insp.GetValue()) / max(total_mass, 1e-30), 0.0))
        if t >= min_time and v < rms_target:
            break
    return t, v


# --------------------------------------------------------------------------
# Run bookkeeping
# --------------------------------------------------------------------------

class RunLog:
    def __init__(self, out_dir: Path, columns: list[str]):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.columns = columns
        self.rows: list[dict] = []
        self.events: list[dict] = []
        self.t_wall0 = time.time()

    def add(self, **row):
        self.rows.append(row)

    def event(self, **ev):
        self.events.append(ev)

    def column(self, name):
        return np.asarray([r[name] for r in self.rows], dtype=float)

    def save(self, summary: dict, meta: dict):
        with open(self.out_dir / "timeseries.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.columns)
            w.writeheader()
            for r in self.rows:
                w.writerow({k: r.get(k, "") for k in self.columns})
        if self.events:
            keys = sorted({k for e in self.events for k in e})
            with open(self.out_dir / "events.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.events)
        summary = dict(summary)
        summary["wall_clock_s"] = round(time.time() - self.t_wall0, 1)
        summary["host"] = platform.node()
        with open(self.out_dir / "summary.json", "w") as f:
            json.dump({"meta": meta, "summary": summary}, f, indent=1, default=_json_default)
        return summary


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def linear_rate(t, m, t0, t1):
    """Least-squares slope of m(t) for t0 <= t <= t1 (kg/s); NaN if fewer than 3 points."""
    t, m = np.asarray(t, float), np.asarray(m, float)
    sel = (t >= t0) & (t <= t1)
    if sel.sum() < 3:
        return float("nan")
    A = np.vstack([t[sel], np.ones(sel.sum())]).T
    slope, _ = np.linalg.lstsq(A, m[sel], rcond=None)[0]
    return float(slope)


def write_states_csv(path, pos, vel, fam, mask):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "vx", "vy", "vz", "family"])
        for p, v, fa in zip(pos[mask], vel[mask], fam[mask]):
            w.writerow([f"{p[0]:.6e}", f"{p[1]:.6e}", f"{p[2]:.6e}", f"{v[0]:.5e}", f"{v[1]:.5e}", f"{v[2]:.5e}",
                        int(fa)])


def mean_mass(templates: list[ClumpTemplate]) -> float:
    return float(np.mean([t.mass for t in templates]))


def deg(x):
    return math.degrees(x)
