"""Open-chute flow at one roundness level and one chute angle (Phase B).

Frame: x down the slope along the floor, y across, z normal to the floor (floor at z = 0).
The chute angle theta is set by tilting gravity: g = g0 * (sin theta, 0, -cos theta).
Grains are fed at a metered rate at the upper end. The run is scored "flows" if, over the
score window, the exit rate keeps up with the feed and the feed zone never backs up.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import geometry as geo
from .config import FAM_PARTICLE, FAM_WALL, cohesion_coefficient, gravity_value
from .sim import (RunLog, common_solver_settings, freeze_rule, grid_points, linear_rate, load_clump_types,
                  load_level, make_solver, particle_states, setup_families, write_states_csv)


def run_chute(cfg, *, level, angle, gravity, seed, out_dir, bond=0.0, device=0, frames=False, run_time=None,
              feed_rate=None, meta_extra=None):
    ch = cfg["chute"]
    L, W = ch["length_m"], ch["width_m"]
    g0 = gravity_value(cfg, gravity)
    th = math.radians(angle)
    gvec = [g0 * math.sin(th), 0.0, -g0 * math.cos(th)]
    C = cohesion_coefficient(bond, g0) if bond > 0 else None
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_time = float(run_time if run_time is not None else ch["run_time_s"])
    feed_rate = float(feed_rate if feed_rate is not None else ch["feed_rate_kg_s"])

    d_nom = cfg["particles"]["d_nominal_m"]
    templates = load_level(cfg, level)
    masses = np.array([t.mass for t in templates])
    rb = max(t.bounding_radius for t in templates)
    spacing = 2.05 * rb
    x0, x1 = ch["feed_x_m"]
    z_ins = ch["feed_insert_height_m"]
    slots = grid_points((x0 + 1.1 * rb, x1 - 1.1 * rb), (-0.5 * W + 1.1 * rb, 0.5 * W - 1.1 * rb), (z_ins, z_ins),
                        spacing)
    if len(slots) == 0:
        raise ValueError("feed zone too small for these particles")
    tray_drop, tray_len = ch["tray_drop_m"], ch["tray_length_m"]

    mesh_path = out_dir / "chute_floor.obj"
    geo.chute_floor_mesh(L, W, ch["floor_mesh_cell_m"]).write_obj(mesh_path, "open chute floor")

    s, mats = make_solver(cfg, device, cohesion_C=C)
    clump_types = load_clump_types(s, templates, mats["particle"])
    floor = s.AddWavefrontMeshObject(str(mesh_path), mats["wall"])
    floor.SetFamily(FAM_WALL)
    planes = (([0, -0.5 * W, 0], [0, 1, 0]), ([0, 0.5 * W, 0], [0, -1, 0]),  # side walls (also bound the tray)
              ([0, 0, 0], [1, 0, 0]),                                         # back plate
              ([0, 0, -tray_drop], [0, 0, 1]),                                # tray floor
              ([L + tray_len, 0, 0], [-1, 0, 0]))                             # tray end wall
    for pt, n in planes:
        s.AddBCPlane(pt, n, mats["wall"]).SetFamily(FAM_WALL)
    setup_families(s)
    freeze_rule(s, f"X > {L + 0.01:.6f} && Z < -0.005", ch["freeze_speed_m_s"], g0)

    interval = ch["feed_interval_s"]
    per_interval = feed_rate * interval
    if per_interval / masses.mean() > len(slots):
        raise ValueError(f"feed zone holds {len(slots)} grains per drop but {per_interval / masses.mean():.0f} are "
                         f"needed for {feed_rate * 1e3:.0f} g/s; lower chute.feed_interval_s")
    carry = 0.0

    def make_layer():
        nonlocal carry
        target = per_interval + carry
        n = int(min(len(slots), max(0, round(target / masses.mean()))))
        idx = rng.choice(len(slots), n, replace=False)
        types = rng.integers(0, len(templates), n)
        mass = float(masses[types].sum())
        carry = min(target - mass, per_interval)  # a skipped drop is made up once, not stockpiled
        P = slots[idx] + rng.uniform(-0.04, 0.04, (n, 3)) * spacing
        return [clump_types[i] for i in types], P.tolist(), mass

    types0, P0, fed = make_layer()
    first = s.AddClumps(types0, P0)
    first.SetVel([0.0, 0.0, -ch["feed_insert_speed_m_s"]])
    s.InstructBoxDomainDimension((-0.01, L + tray_len + 0.01), (-0.5 * W - 0.005, 0.5 * W + 0.005),
                                 (-tray_drop - 0.01, 0.12))
    common_solver_settings(s, cfg, gvec)
    m_exit = s.CreateInspector("clump_mass", f"return X > {L:.6f};")
    m_hold = s.CreateInspector("clump_mass", f"return X <= {L:.6f} && Z > -0.003;")
    m_feedzone = s.CreateInspector("clump_mass", f"return X < {x1 + rb:.6f} && Z > {z_ins - 2.5 * rb:.6f};")

    print(f"[chute] {level} angle={angle} g={gravity} seed={seed}: feed {feed_rate * 1e3:.0f} g/s. "
          f"Initializing (first run compiles kernels)...", flush=True)
    s.Initialize()

    log = RunLog(out_dir, ["t", "fed_kg", "exit_kg", "holdup_kg", "blocked", "dropped"])
    xs = ch["station_fraction"] * L
    half_win = 0.5 * ch["station_window_m"]
    w0, w1 = ch["score_window_s"]
    station = []
    blocked_total, blocked_run, stop_reason, streak = 0.0, 0.0, "time", 0
    frames_dir = out_dir / "frames"
    if frames:
        frames_dir.mkdir(exist_ok=True)
    t, k = 0.0, 0
    while t < run_time - 1e-9:
        s.DoDynamicsThenSync(interval)
        k += 1
        t = k * interval
        zone_kg = float(m_feedzone.GetValue())
        # Never drop a layer onto grains still in the zone; but only call the feed "blocked" when
        # at least three grains sit there for two intervals running (a stray bounce does not count).
        streak = streak + 1 if zone_kg > 3.0 * masses.mean() else 0
        blocked = streak >= 2
        if blocked:
            blocked_total += interval
            blocked_run += interval
            if blocked_run >= ch["blocked_time_to_stop_s"]:
                stop_reason = "feed_backed_up"
        else:
            blocked_run = 0.0
        dropped = 0
        if zone_kg <= 0.0:
            types, P, mass = make_layer()
            if types:
                b = s.AddClumps(types, P)
                b.SetVel([0.0, 0.0, -ch["feed_insert_speed_m_s"]])
                s.UpdateClumps()
                fed += mass
                dropped = 1
        log.add(t=round(t, 5), fed_kg=fed, exit_kg=float(m_exit.GetValue()), holdup_kg=float(m_hold.GetValue()),
                blocked=int(blocked), dropped=dropped)
        if w0 <= t <= w1 and k % 2 == 0:
            pos, vel, fam = particle_states(s)
            sel = (fam == FAM_PARTICLE) & (np.abs(pos[:, 0] - xs) <= half_win) & (pos[:, 2] > -0.002)
            if sel.any():
                station.append((t, float(vel[sel, 0].mean()), float(np.percentile(pos[sel, 2], 95) + 0.5 * d_nom),
                                int(sel.sum())))
            if abs(t - 0.5 * (w0 + w1)) < interval and not (out_dir / "snapshot.csv").exists():
                write_states_csv(out_dir / "snapshot.csv", pos, vel, fam, (fam == FAM_PARTICLE) & (pos[:, 0] <= L))
        if frames and k % max(1, int(round(0.5 / interval))) == 0:
            s.WriteSphereFile(str(frames_dir / f"spheres_{k:05d}.csv"))
        if stop_reason != "time":
            break
    if frames:
        s.WaitForPendingOutput()

    ts = log.column("t")
    exit_rate = linear_rate(ts, log.column("exit_kg"), w0, w1)
    feed_actual = linear_rate(ts, log.column("fed_kg"), w0, w1)
    blocked_in_window = float(sum(r["blocked"] for r in log.rows if w0 <= r["t"] <= w1)) * interval
    ratio = exit_rate / feed_actual if feed_actual and feed_actual > 0 else float("nan")
    # A layer is dropped only into an empty feed zone, but "blocked" needs 3+ grains there, so a few
    # stray grains could starve the feed without counting as blocked. Never score a starved feed.
    feed_fraction = feed_actual / feed_rate if np.isfinite(feed_actual) else float("nan")
    feed_ok = bool(np.isfinite(feed_fraction) and feed_fraction >= ch.get("feed_fraction_min", 0.95))
    skipped_in_window = int(sum(1 - r["dropped"] for r in log.rows if w0 <= r["t"] <= w1))
    flows = (stop_reason == "time" and blocked_in_window == 0.0 and feed_ok and np.isfinite(ratio)
             and ratio >= ch["flow_fraction_threshold"])
    st = np.asarray(station) if station else np.zeros((0, 4))
    summary = {
        "flows": bool(flows),
        "stop_reason": stop_reason,
        "exit_rate_kg_s": exit_rate,
        "feed_rate_kg_s": feed_actual,
        "exit_over_feed": ratio,
        "feed_over_target": feed_fraction,
        "feed_ok": feed_ok,
        "skipped_drops_in_window": skipped_in_window,
        "blocked_s": blocked_total,
        "holdup_end_kg": float(log.rows[-1]["holdup_kg"]) if log.rows else 0.0,
        "station_speed_m_s": float(st[:, 1].mean()) if len(st) else float("nan"),
        "station_depth_m": float(st[:, 2].mean()) if len(st) else float("nan"),
        "simulated_s": float(t),
    }
    meta = {"kind": "chute", "level": level, "roundness": cfg["levels"][level], "angle_deg": angle,
            "gravity": gravity, "g_m_s2": g0, "bond": bond, "seed": seed, "feed_rate_target_kg_s": feed_rate,
            **(meta_extra or {})}
    summary = log.save(summary, meta)
    print(f"[chute] done: flows={flows} exit/feed={ratio:.2f} feed/target={feed_fraction:.2f} speed={summary['station_speed_m_s']:.3f} m/s "
          f"({stop_reason}), wall clock {summary['wall_clock_s']} s", flush=True)
    return summary
