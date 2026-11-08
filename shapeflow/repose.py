"""Angle of repose by slump test (calibration of grain friction against the SRT simulant).

A column of grains stands against a gate in a 20 mm slice. The floor is roughened by one layer of
same-level grains, settled and then fixed (FROZEN family), so the heap cannot slide out on a flat
plane. The gate is removed, the column slumps, and the slope of the heap above the layer is fitted.
Run it at 1 g on L0 and adjust particles.mu in config.json until the angle
matches the simulant's measured angle of repose.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import FAM_FROZEN, FAM_GATE, FAM_GATE_OPEN, FAM_PARTICLE, FAM_SLICE, FAM_WALL, gravity_value
from .sim import (RunLog, common_solver_settings, grid_points, load_clump_types, load_level, make_solver,
                  particle_states, settle, setup_families, stage_positions, write_states_csv)


def heap_angle(pos, d, x_max, bin_w=None, z0=0.0):
    """Fit the free-surface slope of a heap from grain centre positions (x along the floor, z up).

    Heights are measured above z0 (the top of the rough floor layer)."""
    bin_w = bin_w or 2.0 * d
    edges = np.arange(0.0, x_max + bin_w, bin_w)
    tops = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (pos[:, 0] >= a) & (pos[:, 0] < b)
        if sel.sum() >= 2:
            tops.append((0.5 * (a + b), float(np.percentile(pos[sel, 2], 98)) + 0.5 * d - z0))
    if len(tops) < 4:
        return float("nan"), np.asarray(tops)
    prof = np.asarray(tops)
    h = prof[:, 1]
    peak = int(np.argmax(h))
    hmax = h[peak]
    # Fit only the sloping flank to the right of the peak, between 15% and 85% of the peak height.
    flank = prof[peak:]
    sel = (flank[:, 1] > 0.15 * hmax) & (flank[:, 1] < 0.85 * hmax)
    if sel.sum() < 3:
        return float("nan"), prof
    slope = np.polyfit(flank[sel, 0], flank[sel, 1], 1)[0]
    return float(math.degrees(math.atan(-slope))), prof


def run_repose(cfg, *, level, seed, out_dir, mu=None, crr=None, device=0, run_time=None, meta_extra=None):
    rp, p = cfg["repose"], cfg["particles"]
    if mu is not None:
        cfg = {**cfg, "particles": {**p, "mu": float(mu)}}
        p = cfg["particles"]
    if crr is not None:
        cfg = {**cfg, "particles": {**p, "Crr": float(crr)}}
        p = cfg["particles"]
    d = p["d_nominal_m"]
    g = gravity_value(cfg, rp["gravity"])
    # Settling and slumping take longer at low gravity (time ~ 1/sqrt(g), speed ~ sqrt(g)): scale every time
    # limit up and every speed threshold down by ts. ts = 1 at Earth gravity, so Earth runs are unchanged.
    ts = math.sqrt(gravity_value(cfg, "earth") / g)
    T, cw, chh, fl = rp["slice_m"], rp["column_width_m"], rp["column_height_m"], rp["floor_length_m"]
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_time = float(run_time if run_time is not None else rp["run_time_s"] * ts)

    templates = load_level(cfg, level)
    rb = max(t.bounding_radius for t in templates)
    spacing = 2.05 * rb
    vol = float(np.mean([t.mass for t in templates])) / p["density_kg_m3"]
    n_target = int(cw * chh * T * 0.58 / vol)
    m = 1.1 * rb  # wall clearance, leaves room for the placement jitter

    def layer(z):
        return grid_points((m, cw - m), (m, T - m), (z, z), spacing)

    def floor_row(z):
        return grid_points((m, fl - m), (m, T - m), (z, z), spacing)

    stage_h = 0.12
    # Rough floor: about one grain layer (floor area / d^2 grains) dropped over the whole floor.
    n_floor = int(round(fl * T / d ** 2))
    floor_pts = stage_positions(floor_row, m + 0.001, n_floor, stage_h, spacing, rng)
    z_top = 0.02 + chh + stage_h + 0.08

    s, mats = make_solver(cfg, device, rough_floor=True)
    clump_types = load_clump_types(s, templates, mats["particle"])
    s.AddBCPlane([0, 0, 0], [0, 0, 1], mats["rough"]).SetFamily(FAM_WALL)        # floor (rough)
    s.AddBCPlane([0, 0, 0], [1, 0, 0], mats["wall"]).SetFamily(FAM_WALL)         # back wall
    s.AddBCPlane([fl, 0, 0], [-1, 0, 0], mats["wall"]).SetFamily(FAM_WALL)       # far end
    for pt, n in (([0, 0, 0], [0, 1, 0]), ([0, T, 0], [0, -1, 0])):
        s.AddBCPlane(pt, n, mats["slice"]).SetFamily(FAM_SLICE)
    gate = s.AddExternalObject()
    gate.AddPlane([cw, 0, 0], [-1, 0, 0], mats["wall"])
    gate.SetFamily(FAM_GATE_OPEN)  # out of contact while the floor layer settles; closed before the fill
    setup_families(s)
    s.AddClumps([clump_types[i] for i in rng.integers(0, len(templates), len(floor_pts))], floor_pts.tolist())
    s.InstructBoxDomainDimension((-0.01, fl + 0.01), (-0.01, T + 0.01), (-0.01, z_top + 0.02))
    common_solver_settings(s, cfg, [0, 0, -g])
    ke = s.CreateInspector("clump_kinetic_energy")
    m_all = s.CreateInspector("clump_mass")
    z_max = s.CreateInspector("clump_max_z")
    print(f"[repose] {level} mu={p['mu']} Crr={p['Crr']} seed={seed}: {len(floor_pts)} floor-layer grains, then {n_target} "
          f"column grains. Initializing...", flush=True)
    s.Initialize()
    sync = cfg["solver"]["sync_interval_s"]
    # 1. Settle the floor layer, fix it in place, close the gate.
    floor_mass = float(m_all.GetValue())
    settle_log = []

    def logged_settle(phase, mass, t_min, t_max, v_target):
        """settle() plus a record of whether the speed threshold was reached or the time cap was hit."""
        t, v = settle(s, ke, mass, sync, t_min, t_max, v_target)
        settle_log.append({"phase": phase, "sim_s": round(t, 4), "rms_m_s": v, "target_m_s": v_target,
                           "cap_s": t_max, "reached": bool(v < v_target)})
        return t, v

    logged_settle("floor", floor_mass, 0.15 * ts, 1.0 * ts, 0.01 / ts)
    s.ChangeFamily(FAM_PARTICLE, FAM_FROZEN)
    s.ChangeFamily(FAM_GATE_OPEN, FAM_GATE)
    s.DoDynamicsThenSync(sync)  # apply the family changes before reading positions
    pos, _, fam = particle_states(s)
    zf = pos[fam == FAM_FROZEN, 2]
    z_floor = float(np.percentile(zf, 90)) + 0.5 * d  # top of the rough layer (heap heights are measured from it)
    # Some floor grains rest on others, and a grain reaches up to rb from its centre: start the column
    # clear of the highest one, or new grains are created overlapping the layer.
    z_fill = float(zf.max()) + 2.5 * rb
    print(f"[repose] floor layer fixed: {len(zf)} grains, top at {z_floor * 1e3:.1f} mm "
          f"(highest centre {zf.max() * 1e3:.1f} mm; column starts at {z_fill * 1e3:.1f} mm)", flush=True)

    # 2. Fill the column above the layer in stages.
    first = stage_positions(layer, z_fill, n_target, stage_h, spacing, rng)
    s.AddClumps([clump_types[i] for i in rng.integers(0, len(templates), len(first))], first.tolist())
    s.UpdateClumps()
    placed = len(first)
    while True:
        logged_settle(f"fill stage {len(settle_log)}", float(m_all.GetValue()) - floor_mass, 0.15 * ts, 0.6 * ts,
                      0.05 / ts)
        if placed >= n_target:
            break
        block = stage_positions(layer, float(z_max.GetValue()) + 1.5 * rb, n_target - placed, stage_h, spacing, rng)
        if len(block) == 0:
            break
        s.AddClumps([clump_types[i] for i in rng.integers(0, len(templates), len(block))], block.tolist())
        s.UpdateClumps()
        placed += len(block)
    total = float(m_all.GetValue()) - floor_mass  # column grains only (the fixed layer has no kinetic energy)
    _, v_gate = logged_settle("final settle", total, 0.3 * ts, 3.0 * ts, 0.01 / ts)
    fills = [r for r in settle_log if r["phase"].startswith("fill")]
    print(f"[repose] settle: fill stages reached threshold {sum(r['reached'] for r in fills)}/{len(fills)}; final "
          f"settle {'reached' if settle_log[-1]['reached'] else 'TIMED OUT'}; rms at gate {v_gate:.4f} m/s",
          flush=True)
    s.ChangeFamily(FAM_GATE, FAM_GATE_OPEN)
    log = RunLog(out_dir, ["t", "rms_speed_m_s"])
    t = 0.0
    while t < run_time - 1e-9:
        s.DoDynamicsThenSync(sync)
        t += sync
        v = math.sqrt(2 * float(ke.GetValue()) / total)
        log.add(t=round(t, 4), rms_speed_m_s=v)
        if t > 1.0 * ts and v < 0.002 / ts:
            break
    pos, vel, fam = particle_states(s)
    sel = fam == FAM_PARTICLE
    write_states_csv(out_dir / "heap.csv", pos, vel, fam, sel)
    angle, prof = heap_angle(pos[sel], d, fl, z0=z_floor)
    np.savetxt(out_dir / "heap_profile.csv", prof, delimiter=",", header="x_m,top_z_m", comments="")
    summary = {"angle_of_repose_deg": angle, "n_grains": int(placed), "floor_layer_grains": int(len(zf)),
               "time_scale": ts, "settle_log": settle_log, "rms_at_gate_m_s": v_gate,
               "fill_stages_timed_out": sum(not r["reached"] for r in fills),
               "final_settle_timed_out": not settle_log[-1]["reached"],
               "floor_top_m": z_floor, "simulated_s": t,
               "target_deg": rp.get("target_angle_deg")}
    meta = {"kind": "repose", "level": level, "mu": p["mu"], "Crr": p["Crr"], "seed": seed, "gravity": rp["gravity"],
            **(meta_extra or {})}
    summary = log.save(summary, meta)
    print(f"[repose] angle of repose {angle:.1f} deg (mu={p['mu']}, Crr={p['Crr']})", flush=True)
    return summary
