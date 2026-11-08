"""Calibrate grain friction: angle of repose of a roundness level at 1 g (slump test).

Set repose.target_angle_deg in config.json to the SRT simulant's measured angle of repose,
then sweep friction on L0 and pick the mu whose angle matches:

    python scripts/03_repose_calibration.py --level L0 --mu 0.4 0.5 0.6 0.7

Each value runs in its own process (fresh GPU state). Put the chosen value in particles.mu.
Rolling resistance is swept the same way at a fixed mu (put the result in particles.Crr):

    python scripts/03_repose_calibration.py --level L0 --mu 0.6 --crr 0.1 0.2 0.3

Running every level at the chosen mu and Crr gives the repose cross-check in the plan:

    python scripts/03_repose_calibration.py --all-levels

Run folders are named by the values used, e.g. results/repose/L0_mu0p6_crr0p2_s1.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shapeflow.config import load_config, results_dir  # noqa: E402


def one(args, cfg):
    from shapeflow.repose import run_repose
    mu = args.mu[0] if args.mu else float(cfg["particles"]["mu"])
    crr = args.crr[0] if args.crr else float(cfg["particles"]["Crr"])
    tag = f"mu{mu:g}_crr{crr:g}".replace(".", "p")
    out = results_dir() / "repose" / f"{args.level}_{tag}_s{args.seed}"
    run_repose(cfg, level=args.level, seed=args.seed, out_dir=out, mu=mu, crr=crr,
               device=cfg["solver"]["device"] if args.device is None else args.device,
               meta_extra={"run_id": out.name})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", default="L0")
    ap.add_argument("--mu", type=float, nargs="*", default=None, help="friction values to try")
    ap.add_argument("--crr", type=float, nargs="*", default=None, help="rolling resistance values to try")
    ap.add_argument("--all-levels", action="store_true", help="run every level at the config (or given) mu and Crr")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.single:
        one(args, cfg)
        return
    if args.mu and args.crr and len(args.mu) > 1 and len(args.crr) > 1:
        ap.error("sweep either --mu or --crr, not both")
    fixed = []
    if args.mu and len(args.mu) == 1:
        fixed += ["--mu", str(args.mu[0])]
    if args.crr and len(args.crr) == 1:
        fixed += ["--crr", str(args.crr[0])]
    if args.all_levels:
        jobs = [["--level", lv] + fixed for lv in cfg["levels"]]
    elif args.crr and len(args.crr) > 1:
        jobs = [["--level", args.level, "--crr", str(c)] + (["--mu", str(args.mu[0])] if args.mu else [])
                for c in args.crr]
    elif args.mu and len(args.mu) > 1:
        jobs = [["--level", args.level, "--mu", str(m)] + (["--crr", str(args.crr[0])] if args.crr else [])
                for m in args.mu]
    else:
        jobs = [["--level", args.level] + fixed]
    base = [sys.executable, __file__, "--single", "--seed", str(args.seed)]
    if args.config:
        base += ["--config", args.config]
    if args.device is not None:
        base += ["--device", str(args.device)]
    for j in jobs:
        print("$", " ".join(j), flush=True)
        subprocess.run(base + j, check=False)

    rows = []
    for f in sorted((results_dir() / "repose").glob("*/summary.json")):
        d = json.load(open(f))
        rows.append((d["meta"]["level"], d["meta"]["mu"], d["meta"].get("Crr", 0.0),
                     d["summary"]["angle_of_repose_deg"], f.parent.name))
    target = cfg["repose"].get("target_angle_deg")
    print("\nlevel  mu     Crr    angle_deg   run")
    for lv, mu, crr, ang, name in rows:
        print(f"{lv:5s}  {mu:<5g}  {crr:<5g}  {ang:9.1f}   {name}")
    if target:
        l0 = [r for r in rows if r[0] == "L0"]
        if l0:
            best = min(l0, key=lambda r: abs(r[3] - target))
            print(f"\nTarget {target} deg: closest L0 run is mu = {best[1]}, Crr = {best[2]} ({best[3]:.1f} deg).")


if __name__ == "__main__":
    main()
