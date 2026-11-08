"""Design-chart surface: angle of repose over measured roundness R and aspect ratio A, fitted to the Earth grid.

    python scripts/11_design_fit.py

Data: results/repose_grid/grid_results.csv, Earth rows of the 12 grid shapes (seed mean; L0 is not used).
R = measured roundness (mean corner r / r_in), A = measured aspect ratio (b/a + c/a) / 2 (the contour chart's axes).
Candidate forms (least squares):
  linear      angle = c0 + c1 R + c2 A
  linear+R2   angle = c0 + c1 R + c2 A + c3 R^2
  R(1-A)      angle = c0 + c1 R (1 - A)
For each: coefficients, in-sample RMSE, leave-one-out RMSE (refit without each shape, predict it) and the worst LOO
miss. The form with the lowest LOO RMSE is kept; results go to results/repose_grid/design_fit.json.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shapeflow.config import results_dir  # noqa: E402

GRID = results_dir() / "repose_grid"

FORMS = {
    "linear": (["c0", "c_R", "c_A"], lambda R, A: np.c_[np.ones_like(R), R, A]),
    "linear+R2": (["c0", "c_R", "c_A", "c_R2"], lambda R, A: np.c_[np.ones_like(R), R, A, R ** 2]),
    "R(1-A)": (["c0", "c_R(1-A)"], lambda R, A: np.c_[np.ones_like(R), R * (1 - A)]),
}


def main():
    rows = [r for r in csv.DictReader(open(GRID / "grid_results.csv"))
            if r["gravity"] == "earth" and r["shape"].startswith("R")]
    names = [r["shape"] for r in rows]
    R = np.array([float(r["roundness_measured"]) for r in rows])
    A = np.array([0.5 * (float(r["b_a"]) + float(r["c_a"])) for r in rows])
    y = np.array([float(r["mean"]) for r in rows])
    n_seeds = {r["shape"]: int(r["n_seeds"]) for r in rows}
    print(f"{len(rows)} shapes: " + ", ".join(f"{s} {v:.2f}{' (1 seed)' if n_seeds[s] == 1 else ''}"
                                               for s, v in zip(names, y)))
    out = {"n_shapes": len(rows), "data": [{"shape": s, "R": round(float(a), 3), "A": round(float(b), 3),
                                            "angle_mean": float(v), "n_seeds": n_seeds[s]}
                                           for s, a, b, v in zip(names, R, A, y)], "forms": {}}
    for form, (cn, X) in FORMS.items():
        M = X(R, A)
        c, *_ = np.linalg.lstsq(M, y, rcond=None)
        rmse = float(np.sqrt(np.mean((M @ c - y) ** 2)))
        loo = []
        for i in range(len(y)):
            k = np.arange(len(y)) != i
            ci, *_ = np.linalg.lstsq(M[k], y[k], rcond=None)
            loo.append(float(M[i] @ ci - y[i]))
        loo = np.array(loo)
        w = int(np.argmax(np.abs(loo)))
        out["forms"][form] = {"coefficients": dict(zip(cn, np.round(c, 3).tolist())), "rmse_in_sample": round(rmse, 3),
                              "rmse_loo": round(float(np.sqrt(np.mean(loo ** 2))), 3),
                              "worst_loo": {"shape": names[w], "predicted_minus_measured": round(float(loo[w]), 2)},
                              "loo_misses": dict(zip(names, np.round(loo, 2).tolist()))}
        f = out["forms"][form]
        print(f"\n{form}: " + ", ".join(f"{k} {v:+.3f}" for k, v in f["coefficients"].items()))
        print(f"  in-sample RMSE {f['rmse_in_sample']:.2f} deg; LOO RMSE {f['rmse_loo']:.2f} deg; worst LOO miss "
              f"{names[w]} {loo[w]:+.2f} deg (predicted - measured)")
    best = min(out["forms"], key=lambda k: out["forms"][k]["rmse_loo"])
    out["kept"] = best
    print(f"\nkept: {best} (lowest LOO RMSE, {out['forms'][best]['rmse_loo']:.2f} deg)")
    json.dump(out, open(GRID / "design_fit.json", "w"), indent=1)


if __name__ == "__main__":
    main()
