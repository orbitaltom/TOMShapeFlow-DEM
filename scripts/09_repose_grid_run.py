"""One repose run of the shape grid (same slump test as the calibration).

    python scripts/09_repose_grid_run.py --shape R013_A08 --seed 1 --frames
    python scripts/09_repose_grid_run.py --shape R045_A08 --seed 1 --gravity lunar
    python scripts/09_repose_grid_run.py --shape R013_A06 --seed 1 --frames --dt 2.5e-6   # memory fallback

Templates come from shapes/repose_grid/templates/ (scripts/08_repose_grid_shapes.py); output goes to
results/repose_grid/runs/<shape>_<gravity>_s<seed>/. mu and Crr are the config values. A shape that is not in the
grid templates (e.g. L0, the calibration anchor) is taken from the shipped shapes/templates/.
--frames saves grain positions, orientations (quaternions) and velocities every 0.02 s from gate opening until
the grains stop (frames/frame_NNNN.npz, plus frames/meta.npz with the template index and family of each grain).
Every run writes progress.csv (wall clock and simulated time at the start and end of each solver call), which the
queue watchdog reads to get ms/step while the run is going.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shapeflow.repose as rp  # noqa: E402
from shapeflow.config import FAM_FROZEN, FAM_GATE, FAM_GATE_OPEN, FAM_PARTICLE, load_config, results_dir  # noqa: E402

GRID = results_dir() / "repose_grid"
FRAME_DT = 0.02


class Heartbeat:
    """Wraps the DEME solver: logs the start and end of every DoDynamicsThenSync call to progress.csv, so a watchdog
    can tell ms/step during the run (including a lower bound while a long call is still in progress). The calls
    themselves are unchanged."""

    def __init__(self, s, path, dt):
        self._s, self.dt = s, dt
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", buffering=1)
        self.f.write("event,wall_s,sim_s,call_s,dt_s\n")

    def __getattr__(self, name):
        return getattr(self._s, name)

    def DoDynamicsThenSync(self, T):
        self.f.write(f"start,{time.time():.3f},{float(self._s.GetSimTime()):.6f},{T:.6g},{self.dt:g}\n")
        r = self._s.DoDynamicsThenSync(T)
        self.f.write(f"end,{time.time():.3f},{float(self._s.GetSimTime()):.6f},{T:.6g},{self.dt:g}\n")
        return r


class FrameRecorder:
    """Wraps the DEME solver: records template indices of added clumps, then saves frames during the slump."""

    def __init__(self, s, frames_dir):
        self._s, self.dir = s, frames_dir
        self.type_ids, self.templates = {}, []
        self.slump, self.t_gate, self.k = False, None, 0
        self.order_checks = []
        self._last_batch = None

    def __getattr__(self, name):
        return getattr(self._s, name)

    def LoadClumpType(self, *a, **kw):
        ct = self._s.LoadClumpType(*a, **kw)
        self.type_ids[id(ct)] = len(self.type_ids)
        return ct

    def AddClumps(self, types, pos):
        self.templates.extend(self.type_ids[id(t)] for t in types)
        self._last_batch = np.asarray(pos, float)
        return self._s.AddClumps(types, pos)

    def _states(self):
        n = int(self._s.GetNumOwners())
        fam = np.asarray(self._s.GetOwnerFamily(0, n), int).reshape(-1)
        sel = np.where((fam == FAM_PARTICLE) | (fam == FAM_FROZEN))[0]
        pos = np.asarray(self._s.GetOwnerPosition(0, n), float).reshape(-1, 3)[sel]
        q = np.asarray(self._s.GetOwnerOriQ(0, n), float).reshape(-1, 4)[sel]
        v = np.asarray(self._s.GetOwnerVelocity(0, n), float).reshape(-1, 3)[sel]
        return sel, fam[sel], pos, q, v

    def UpdateClumps(self):
        self._s.UpdateClumps()
        # Check that clump owners are in insertion order: the newest owners sit where the batch was placed.
        _, _, pos, _, _ = self._states()
        b = self._last_batch
        ok = len(pos) == len(self.templates) and np.allclose(pos[-len(b):], b, atol=1e-6)
        self.order_checks.append(bool(ok))

    def ChangeFamily(self, a, b):
        self._s.ChangeFamily(a, b)
        if a == FAM_GATE and b == FAM_GATE_OPEN:
            self.slump, self.t_gate = True, float(self._s.GetSimTime())
            self.dir.mkdir(parents=True, exist_ok=True)
            self._save()

    def _save(self):
        sel, fam, pos, q, v = self._states()
        if self.k == 0:
            np.savez_compressed(self.dir / "meta.npz", template=np.asarray(self.templates, np.int16),
                                family=fam.astype(np.int8), order_checks=np.asarray(self.order_checks, bool),
                                frame_dt=FRAME_DT)
        np.savez_compressed(self.dir / f"frame_{self.k:04d}.npz", t=float(self._s.GetSimTime()) - self.t_gate,
                            pos=pos.astype(np.float32), quat=q.astype(np.float32), vel=v.astype(np.float32))
        self.k += 1

    def DoDynamicsThenSync(self, T):
        if not self.slump:
            return self._s.DoDynamicsThenSync(T)
        t_end = float(self._s.GetSimTime()) + T
        while True:
            t = float(self._s.GetSimTime())
            nxt = self.t_gate + FRAME_DT * (self.k)  # time of the next frame to save
            if nxt <= t + 1e-9:
                self._save()
                continue
            step = min(nxt, t_end) - t
            if step <= 1e-9:
                break
            self._s.DoDynamicsThenSync(step)
            if abs(float(self._s.GetSimTime()) - nxt) < 1e-7:
                self._save()
            if float(self._s.GetSimTime()) >= t_end - 1e-9:
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gravity", default="earth", choices=["earth", "lunar"])
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--dt", type=float, default=None,
                    help="override solver.time_step_s (run folder gets a _dt<value> suffix), e.g. 2.5e-6")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config()
    # Grid shapes come from the grid templates; anything else (e.g. L0, the calibration anchor) from the shipped
    # templates in shapes/templates/.
    grid_idx = Path(__file__).resolve().parent.parent / "shapes" / "repose_grid" / "templates" / "templates_index.json"
    in_grid = grid_idx.exists() and args.shape in json.load(open(grid_idx))
    cfg["shapes"]["templates_dir"] = "shapes/repose_grid/templates" if in_grid else cfg["shapes"]["templates_dir"]
    cfg["repose"]["gravity"] = args.gravity
    if args.dt:
        cfg["solver"]["time_step_s"] = args.dt
    tag = f"_dt{args.dt:g}" if args.dt else ""
    out = GRID / "runs" / f"{args.shape}_{args.gravity}{tag}_s{args.seed}"
    rec = {}
    orig = rp.make_solver

    def make_solver(*a, **kw):
        s, mats = orig(*a, **kw)
        s = Heartbeat(s, out / "progress.csv", cfg["solver"]["time_step_s"])
        if args.frames:
            s = FrameRecorder(s, out / "frames")
            rec["r"] = s
        return s, mats
    rp.make_solver = make_solver
    t0 = time.time()
    summ = rp.run_repose(cfg, level=args.shape, seed=args.seed, out_dir=out, mu=cfg["particles"]["mu"],
                         crr=cfg["particles"]["Crr"], device=args.device,
                         meta_extra={"run_id": out.name, "shape": args.shape, "study": "repose_grid",
                                     "time_step_s": cfg["solver"]["time_step_s"]})
    if args.frames:
        r = rec["r"]
        r._save()  # final state (the heap pictures use it)
        sz = sum(f.stat().st_size for f in (out / "frames").glob("*.npz"))
        info = {"frames": r.k, "frame_dt": FRAME_DT, "frames_MB": round(sz / 1e6, 1),
                "order_checks_all_ok": all(r.order_checks), "n_order_checks": len(r.order_checks)}
        json.dump(info, open(out / "frames_info.json", "w"), indent=1)
        print(f"[grid] frames: {info}", flush=True)
    print(f"[grid] {args.shape} {args.gravity} seed {args.seed}: angle {summ['angle_of_repose_deg']:.2f} deg, "
          f"process {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
