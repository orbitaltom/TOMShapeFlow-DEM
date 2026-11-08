"""Make the clump templates for the repose shape grid (CPU only).

    python scripts/08_repose_grid_shapes.py            # all 12 shapes -> shapes/repose_grid/templates/
    python scripts/08_repose_grid_shapes.py --only R013_A08

Roundness 0.13 / 0.45 / 0.80 / 1.0 x aspect b/a = c/a 0.6 / 0.8 / 1.0, 8 templates each, same volumes as the
shipped levels (d_v 1.8-2.2 mm). Aspect within +-0.03 per template, measured roundness mean within +-0.05 per shape
(design roundness tuned in rounds). Bump tolerance 3.5% of d for roundness 0.13, 2% otherwise. Writes the templates,
templates_index.json and shapes_summary.csv (shapes/repose_grid/).
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shapeflow import grid_shapes as gs  # noqa: E402
from shapeflow.config import PKG_ROOT, load_config  # noqa: E402

OUT = PKG_ROOT / "shapes" / "repose_grid" / "templates"
# Starting design roundness per target (measured/target ratios from the 2-template pricing test).
R_START = {0.13: 0.13 * 1.34, 0.45: 0.45 * 1.22, 0.80: 0.80 * 1.03}


def _one(args):
    key, R, R_d, ar, k, d_t, rho, seed = args
    t, extra = gs.polyhedral_template(key, R, R_d, ar, k, d_t, rho, seed)
    return k, t, extra


def build_polyhedral(key, R, ar, d_targets, rho, seed, pool, log):
    R_d = R_START[R]
    for rnd in range(5):
        # Fill all 8 template slots at this design roundness; keep accepted ones and retry only the failures.
        got = {}
        for attempt in range(8):
            todo = [k for k in range(len(d_targets)) if k not in got]
            if not todo:
                break
            jobs = [(key, R, R_d, ar, k, d_targets[k], rho, seed + 1000 * rnd + 97 * attempt) for k in todo]
            for k, t, e in pool.map(_one, jobs):
                if t is not None:
                    got[k] = (k, t, e)
            if len(got) < len(d_targets):
                log(f"  {key} round {rnd}, attempt {attempt}: {len(d_targets) - len(got)} templates outside the aspect "
                    f"window; retrying those")
        if len(got) < len(d_targets):
            raise RuntimeError(f"{key}: could not fill all templates within the aspect window")
        res = [got[k] for k in sorted(got)]
        Rm = float(np.mean([t.corner_radius_ratio for _, t, _ in res]))
        log(f"  {key} round {rnd}: R_design {R_d:.3f} -> measured mean {Rm:.3f} (target {R:.2f}), "
            f"spheres {[t.n_spheres for _, t, _ in res]}")
        if abs(Rm - R) <= gs.ROUND_TOL:
            return [(t, e) for _, t, e in res]
        R_d = float(np.clip(R_d * R / max(Rm, 1e-3), 0.02, 0.98))
    raise RuntimeError(f"{key}: roundness not within tolerance after 5 rounds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config()
    p, s = cfg["particles"], cfg["shapes"]
    d_targets = np.linspace(p["d_min_m"], p["d_max_m"], s["n_templates_per_level"])
    rho = p["density_kg_m3"]
    OUT.mkdir(parents=True, exist_ok=True)
    idx_path = OUT / "templates_index.json"
    index = json.load(open(idx_path)) if idx_path.exists() else {}
    log = lambda m: print(m, flush=True)  # noqa: E731
    with ProcessPoolExecutor(args.workers) as pool:
        for ar in gs.GRID_AR:
            for R in gs.GRID_R:
                key = gs.shape_key(R, ar)
                if args.only and key not in args.only:
                    continue
                log(f"{key}: roundness {R}, aspect {ar}, bump tol {100 * gs.bump_tol(R):.1f}% of d")
                if R >= 0.999 and ar >= 0.999:
                    temps = gs.sphere_templates(key, d_targets, rho)
                elif R >= 0.999:
                    temps = gs.capsule_templates(key, ar, d_targets, rho)
                else:
                    temps = build_polyhedral(key, R, ar, d_targets, rho, s["seed"], pool, log)
                index[key] = []
                for t, extra in temps:
                    fname = f"{key}_t{t.index}.csv"
                    with open(OUT / fname, "w") as f:
                        f.write("x,y,z,r\n")
                        for q, r in zip(t.positions_m, t.radii_m):
                            f.write(f"{q[0]:.9e},{q[1]:.9e},{q[2]:.9e},{r:.9e}\n")
                    meta = gs.template_meta(t, extra)
                    meta.update({"file": fname, "roundness_target": R, "aspect_target": ar})
                    index[key].append(meta)
                json.dump(index, open(idx_path, "w"), indent=1)
    rows = []
    for key, ts in index.items():
        m = lambda f: float(np.mean([t[f] for t in ts]))  # noqa: E731
        rows.append({"shape": key, "roundness_target": ts[0]["roundness_target"], "aspect_target": ts[0]["aspect_target"],
                     "bump_tol_pct_d": round(ts[0]["tol_pct_d"], 1), "R_design": round(ts[0]["R_design"], 3),
                     "roundness_measured_mean": round(m("corner_radius_ratio"), 3),
                     "b_a_mean": round(m("aspect_b_a"), 3), "c_a_mean": round(m("aspect_c_a"), 3),
                     "sphericity_mean": round(m("sphericity"), 3), "spheres_mean": round(m("n_spheres"), 1),
                     "spheres_max": max(t["n_spheres"] for t in ts),
                     "bump97_pct_d_mean": round(m("bump97_pct_d"), 2),
                     "bumpmax_pct_d_max": round(max(t["bumpmax_pct_d"] for t in ts), 2),
                     "mass_mean_mg": round(m("mass_kg") * 1e6, 2)})
    rows.sort(key=lambda r: (r["aspect_target"], r["roundness_target"]))
    with open(OUT.parent / "shapes_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
