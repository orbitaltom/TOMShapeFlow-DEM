"""Results table of the repose shape grid: results/repose_grid/grid_results.csv.

    python scripts/12_grid_table.py

One row per shape and gravity (Earth: the 12 grid shapes and the L0 calibration anchor; lunar: the shapes that were
run at lunar g), built from results/repose_grid/runs/<shape>_<gravity>[_dt2.5e-06]_s<seed>/summary.json and the
template descriptors in shapes/repose_grid/shapes_summary.csv.

Time step per seed: a run folder with the _dt2.5e-06 suffix was run at dt 2.5e-6, otherwise 5e-6. For an Earth
shape that has dt 2.5e-6 runs, those are used for both seeds (the whole shape was rerun); otherwise each seed uses
its dt 2.5e-6 run if there is one. time_step_s reads "mixed" when the two seeds differ.

Settle diagnostics per seed: logged in summary.json (settle_log) for runs made after settle logging was added; for
earlier runs the rms speed at gate opening is inferred from the first slump frame (frames/, in the separate data
bundle), and the final settle counts as timed out if that rms is at or above its speed threshold.

Lunar minus Earth: sd = 1.14 x sqrt(1/n_lunar + 1/n_earth) deg, i.e. 1.4 deg for one lunar seed against a 2-seed
Earth mean (per-seed sd 1.4 / sqrt(1.5) = 1.143 deg).
"""

import csv
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shapeflow.config import results_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRID = results_dir() / "repose_grid"
RUNS = GRID / "runs"
SHAPES = ROOT / "shapes" / "repose_grid" / "shapes_summary.csv"
ALL = [f"R{r}_A{a}" for a in ("06", "08", "10") for r in ("013", "045", "080", "100")]
FB = "_dt2.5e-06"
SIG_SEED = 1.4 / 1.5 ** 0.5


def summary(shape, seed, g, tag=""):
    f = RUNS / f"{shape}_{g}{tag}_s{seed}" / "summary.json"
    return json.load(open(f))["summary"] if f.exists() else None


def settle_info(shape, seed, g, tag):
    """(rms at gate m/s, final settle timed out, fill stages timed out, source) for one run."""
    s = summary(shape, seed, g, tag)
    if s and "final_settle_timed_out" in s:
        return round(s["rms_at_gate_m_s"], 4), bool(s["final_settle_timed_out"]), s["fill_stages_timed_out"], "logged"
    fd = RUNS / f"{shape}_{g}{tag}_s{seed}" / "frames"
    if s and (fd / "frame_0000.npz").exists() and (fd / "meta.npz").exists():
        free = np.load(fd / "meta.npz")["family"] == 0
        v = np.load(fd / "frame_0000.npz")["vel"][free].astype(float)
        rms = float(np.sqrt((v ** 2).sum(1).mean()))
        ts = s.get("time_scale", 1.0)
        # the final settle only ends early once rms < 0.01/ts, so rms >= that at gate opening means it hit its cap
        return round(rms, 4), bool(rms >= 0.01 / ts), "", "inferred from gate frame (translational rms)"
    return "", "", "", "not recorded" if s else ""


def main():
    shp = {r["shape"]: r for r in csv.DictReader(open(SHAPES))}
    rows = []
    for sh in ALL + ["L0"]:
        for g in ("earth", "lunar"):
            fb_shape = g == "earth" and any(summary(sh, s_, g, FB) for s_ in (1, 2))
            tags = {s_: FB if (fb_shape or summary(sh, s_, g, FB)) else "" for s_ in (1, 2)}
            angs = {s_: summary(sh, s_, g, tags[s_]) for s_ in (1, 2)}
            angs = {s_: v["angle_of_repose_deg"] for s_, v in angs.items() if v}
            if not angs:
                continue
            dts = {s_: (2.5e-6 if tags[s_] else 5e-6) for s_ in angs}
            m = shp.get(sh, {})
            v = list(angs.values())
            row = {"shape": sh, "gravity": g, "roundness_target": m.get("roundness_target"),
                   "roundness_measured": m.get("roundness_measured_mean"), "aspect_target": m.get("aspect_target"),
                   "b_a": m.get("b_a_mean"), "c_a": m.get("c_a_mean"), "sphericity": m.get("sphericity_mean"),
                   "bump_tol_pct_d": m.get("bump_tol_pct_d"), "spheres_mean": m.get("spheres_mean"),
                   "angle_s1": round(angs[1], 2) if 1 in angs else "", "angle_s2": round(angs[2], 2) if 2 in angs else "",
                   "mean": round(st.mean(v), 2), "spread": round(max(v) - min(v), 2) if len(v) > 1 else "",
                   "time_step_s": (dts[min(dts)] if len(set(dts.values())) == 1 else "mixed"),
                   "time_step_s1": dts.get(1, ""), "time_step_s2": dts.get(2, ""), "n_seeds": len(v),
                   "seeds_note": "1 seed" if (g == "earth" and len(v) == 1) else ""}
            flags = []
            for seed in (1, 2):
                rms, to, fills, src = settle_info(sh, seed, g, tags[seed])
                row.update({f"rms_gate_s{seed}": rms, f"final_settle_timed_out_s{seed}": to,
                            f"fill_stages_timed_out_s{seed}": fills, f"settle_source_s{seed}": src})
                if to is True:
                    flags.append(f"seed {seed}")
            row["settle_flag"] = ("FINAL SETTLE TIMED OUT: " + ", ".join(flags)) if flags else ""
            rows.append(row)
    earth = {r["shape"]: r for r in rows if r["gravity"] == "earth"}
    for r in rows:
        e = earth.get(r["shape"]) if r["gravity"] == "lunar" else None
        r["earth_mean"] = e["mean"] if e else ""
        r["lunar_minus_earth"] = round(r["mean"] - e["mean"], 2) if e else ""
        r["lunar_minus_earth_sd"] = round(SIG_SEED * (1 / r["n_seeds"] + 1 / e["n_seeds"]) ** 0.5, 2) if e else ""
    with open(GRID / "grid_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {GRID / 'grid_results.csv'}")


if __name__ == "__main__":
    main()
