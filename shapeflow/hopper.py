"""Wedge-hopper discharge at one roundness level (Phases A, C, D, E).

Frame: x across the hopper, y through the slice (0 .. slice), z up; slot outlet at z = 0.
Sequence: fill under fill gravity -> settle -> switch to run gravity -> relax -> open gate ->
log discharged mass every sync interval. A clog (no discharge for clog_window_s) is logged,
then the grains just above the slot become fixed, non-interacting ghosts so flow can resume.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import geometry as geo
from .config import (FAM_GATE, FAM_GATE_OPEN, FAM_GHOST, FAM_PARTICLE, FAM_SLICE, FAM_WALL, cohesion_coefficient,
                     gravity_value, results_dir)
from .sim import (RunLog, common_solver_settings, contact_multiplicity, freeze_rule, grid_points, linear_rate,
                  load_clump_types, load_level, lookup_multiplicity, make_solver, mean_mass, particle_states, settle,
                  setup_families, stage_positions, write_states_csv)


def hopper_layer(z, top_width, outlet, theta, slice_depth, spacing, rb):
    """Grid centres of one horizontal layer at height z, clear of the walls by at least rb."""
    hw = geo.hopper_half_width(z - rb, top_width, outlet, theta) - 1.1 * rb
    if hw <= 0:
        return np.zeros((0, 3))
    layer = grid_points((-hw, hw), (1.1 * rb, slice_depth - 1.1 * rb), (z, z), spacing)
    if len(layer):
        layer[:, 0] -= 0.5 * (layer[:, 0].max() + layer[:, 0].min())  # centre the row
    return layer


def run_hopper(cfg, *, level, outlet_mult, gravity, bond, seed, mode, out_dir, n_particles, theta=None,
               device=0, frames=False, run_time=None, meta_extra=None):
    hp, p = cfg["hopper"], cfg["particles"]
    d = p["d_nominal_m"]
    D = outlet_mult * d
    theta = float(theta if theta is not None else hp["wall_angle_from_vertical_deg"])
    T, W = hp["slice_m"], hp["top_width_m"]
    g_run = gravity_value(cfg, gravity)
    g_fill = gravity_value(cfg, hp["fill_gravity"])
    C, n_mult_used = None, None
    if bond > 0:
        # DEME pulls once per touching sphere pair; divide by the measured pairs per grain contact.
        n_mult_used = lookup_multiplicity(cfg, level, results_dir())
        C = cohesion_coefficient(bond, g_run) / n_mult_used
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    templates = load_level(cfg, level)
    m_mean = mean_mass(templates)
    rb = max(t.bounding_radius for t in templates)
    spacing = 2.05 * rb
    stage_h = hp["fill_stage_height_m"]

    def layer(z):
        return hopper_layer(z, W, D, theta, T, spacing, rb)

    first = stage_positions(layer, rb + 0.002, n_particles, stage_h, spacing, rng)
    # Walls must stand above the highest stage that can be dropped in.
    z_top = hp["nominal_wall_height_m"] + stage_h + 0.05
    z_m = -hp["measure_plane_multiple_of_D"] * D
    zc, xc = hp["catch_depth_m"], hp["catch_halfwidth_m"]

    mesh_path = out_dir / "hopper_walls.obj"
    geo.hopper_wall_mesh(W, D, theta, z_top, T, hp["wall_mesh_cell_m"]).write_obj(
        mesh_path, f"hopper walls theta={theta} D={D}")

    s, mats = make_solver(cfg, device, cohesion_C=C)
    clump_types = load_clump_types(s, templates, mats["particle"])
    walls = s.AddWavefrontMeshObject(str(mesh_path), mats["wall"])
    walls.SetFamily(FAM_WALL)
    for pt, n in (([0, 0, 0], [0, 1, 0]), ([0, T, 0], [0, -1, 0])):
        s.AddBCPlane(pt, n, mats["slice"]).SetFamily(FAM_SLICE)
    gate = s.AddExternalObject()
    gate.AddPlane([0, 0, 0], [0, 0, 1], mats["wall"])
    gate.SetFamily(FAM_GATE)
    for pt, n in (([0, 0, -zc], [0, 0, 1]), ([-xc, 0, 0], [1, 0, 0]), ([xc, 0, 0], [-1, 0, 0])):
        s.AddBCPlane(pt, n, mats["wall"]).SetFamily(FAM_WALL)
    setup_families(s)
    freeze_rule(s, f"Z < {z_m - 0.01:.6f}", hp["freeze_speed_m_s"], g_run)
    s.AddClumps([clump_types[i] for i in rng.integers(0, len(templates), len(first))], first.tolist())
    s.InstructBoxDomainDimension((-xc - 0.02, xc + 0.02), (-0.01, T + 0.01), (-zc - 0.02, z_top + 0.05))
    common_solver_settings(s, cfg, [0, 0, -g_fill])
    ke = s.CreateInspector("clump_kinetic_energy")
    m_all = s.CreateInspector("clump_mass")
    z_max = s.CreateInspector("clump_max_z")
    m_out = s.CreateInspector("clump_mass", f"return Z < {z_m:.6f};")
    # Arch-removal box above the slot. Ghosts are fixed, so right after a removal everything in it is a ghost.
    hx = 0.5 * D + hp["arch_box_halfwidth_multiple_of_D"] * D
    z_lo, z_hi = -0.5 * D, hp["arch_box_height_multiple_of_D"] * D
    m_box = s.CreateInspector("clump_mass", f"return X > {-hx:.6f} && X < {hx:.6f} && Z > {z_lo:.6f} && Z < {z_hi:.6f};")

    print(f"[hopper] {level} theta={theta} D={outlet_mult}d g={gravity} Bo={bond} seed={seed}: filling "
          f"{n_particles} grains in stages. Initializing (first run compiles kernels)...", flush=True)
    s.Initialize()
    sync = cfg["solver"]["sync_interval_s"]

    # 1. Fill in stages under fill gravity: drop a loose block onto the bed, let it land, repeat.
    placed, t_fill, stages = len(first), 0.0, 1
    while True:
        dt, _ = settle(s, ke, float(m_all.GetValue()), sync, min(0.15, hp["fill_stage_settle_s"]),
                       hp["fill_stage_settle_s"] * 2, 0.05)
        t_fill += dt
        if placed >= n_particles:
            break
        top = float(z_max.GetValue())
        block = stage_positions(layer, top + 1.5 * rb, n_particles - placed, stage_h, spacing, rng)
        if len(block) == 0:
            raise RuntimeError("no room for the next fill stage; raise hopper.nominal_wall_height_m")
        s.AddClumps([clump_types[i] for i in rng.integers(0, len(templates), len(block))], block.tolist())
        s.UpdateClumps()
        placed += len(block)
        stages += 1
    total = float(m_all.GetValue())
    dt, v_rms = settle(s, ke, total, sync, 0.2, hp["fill_settle_max_s"], hp["fill_settle_rms_speed_m_s"])
    t_fill += dt
    fill_top = float(z_max.GetValue())
    print(f"[hopper] filled in {stages} stages, {t_fill:.2f} s simulated; bed top {fill_top:.3f} m, "
          f"rms speed {v_rms:.3g} m/s", flush=True)
    n_mult = contact_multiplicity(s, out_dir / "contacts_after_fill.csv")
    print(f"[hopper] {n_mult:.2f} touching sphere pairs per touching grain pair", flush=True)

    # 2. Run gravity, relax, open.
    s.SetGravitationalAcceleration([0, 0, -g_run])
    s.DoDynamicsThenSync(hp["relax_time_s"])
    s.ChangeFamily(FAM_GATE, FAM_GATE_OPEN)
    t0 = float(s.GetSimTime())

    # Discharge time scales as 1/sqrt(g), so flow-run times are given for lunar gravity and scaled
    # for other gravities (Earth runs last 0.41x as long, with the same mass discharged).
    tscale = math.sqrt(gravity_value(cfg, "lunar") / g_run)
    if run_time is None:
        run_time = (hp["flow_run_time_s"] if mode == "flow" else hp["clog_run_time_s"]) * tscale
    log = RunLog(out_dir, ["t", "discharged_kg", "ghost_kg", "rms_speed_m_s"])
    ghost_kg, n_clogs = 0.0, 0
    last_m, t_last_flow = 0.0, 0.0
    snap_done = False
    frames_dir = out_dir / "frames"
    if frames:
        frames_dir.mkdir(exist_ok=True)
    t, k = 0.0, 0
    while t < run_time - 1e-9:
        s.DoDynamicsThenSync(sync)
        k += 1
        t = float(s.GetSimTime()) - t0
        m = float(m_out.GetValue())
        v_rms = math.sqrt(max(2.0 * float(ke.GetValue()) / total, 0.0))
        log.add(t=round(t, 5), discharged_kg=m, ghost_kg=ghost_kg, rms_speed_m_s=v_rms)
        if m > last_m + 0.5 * m_mean:
            last_m, t_last_flow = m, t
        if not snap_done and t >= hp["snapshot_time_s"] * tscale:
            pos, vel, fam = particle_states(s)
            write_states_csv(out_dir / "snapshot.csv", pos, vel, fam, (fam == FAM_PARTICLE) & (pos[:, 2] > -D))
            snap_done = True
        if frames and k % max(1, int(round(0.5 / sync))) == 0:
            s.WriteSphereFile(str(frames_dir / f"spheres_{k:05d}.csv"))
        if m >= total - ghost_kg - 2 * m_mean:
            log.event(t=t, kind="empty", discharged_kg=m)
            break
        if t - t_last_flow >= hp["clog_window_s"] * tscale:
            removed = int(s.ChangeClumpFamily(FAM_GHOST, (-hx, hx), (-1.0, 1.0), (z_lo, z_hi), {FAM_PARTICLE}))
            if removed == 0:
                # Nothing left above the slot: the hopper has emptied, not clogged.
                log.event(t=t, kind="empty", discharged_kg=m)
                break
            n_clogs += 1
            ghost_kg = float(m_box.GetValue())
            log.event(t=t, kind="clog", discharged_kg=m, clog_started_t=t_last_flow, ghosts_removed=removed)
            print(f"[hopper] clog {n_clogs}: no discharge since t={t_last_flow:.2f} s (detected at {t:.2f} s); "
                  f"{removed} grains above the slot ghosted", flush=True)
            t_last_flow = t  # restart the clog clock
            if mode == "clog" and n_clogs >= hp["clog_max_events"]:
                break
    if frames:
        s.WaitForPendingOutput()

    ts, ms = log.column("t"), log.column("discharged_kg")
    w0, w1 = (x * tscale for x in hp["flow_window_s"])
    clog_ev = [e for e in log.events if e["kind"] == "clog"]
    # Avalanche = mass discharged between the start (or the previous clog) and the next clog.
    bounds = [0.0] + [e["discharged_kg"] for e in clog_ev]
    avalanches = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
    clog_in_window = any(w0 <= e["clog_started_t"] <= w1 or w0 <= e["t"] <= w1 for e in clog_ev)
    sel = (ts >= w0) & (ts <= w1)
    q_avg = float((ms[sel][-1] - ms[sel][0]) / (ts[sel][-1] - ts[sel][0])) if sel.sum() >= 2 else float("nan")
    summary = {
        # Steady discharge rate; NaN when a clog falls inside the window (see Q_window_avg_kg_s).
        "Q_kg_s": float("nan") if clog_in_window else linear_rate(ts, ms, w0, min(w1, run_time)),
        "Q_window_avg_kg_s": q_avg,
        "clog_in_window": bool(clog_in_window),
        "sphere_pairs_per_contact": n_mult,
        "cohesion_multiplicity_used": n_mult_used,
        "discharged_kg": float(ms[-1]) if len(ms) else 0.0,
        "total_kg": total,
        "ghost_kg": ghost_kg,
        "n_clogs": n_clogs,
        "avalanche_kg": avalanches,
        "clog_free": n_clogs == 0,
        "simulated_s": float(t),
        "fill_settle_s": t_fill,
        "fill_stages": stages,
        "fill_top_m": fill_top,
        "n_grains": int(placed),
    }
    meta = {"kind": "hopper", "level": level, "roundness": cfg["levels"][level], "outlet_mult": outlet_mult,
            "D_m": D, "theta_deg": theta, "gravity": gravity, "g_m_s2": g_run, "bond": bond, "cohesion_C": C,
            "seed": seed, "mode": mode, **(meta_extra or {})}
    summary = log.save(summary, meta)
    print(f"[hopper] done: Q={summary['Q_kg_s'] * 1e3:.2f} g/s, clogs={n_clogs}, "
          f"wall clock {summary['wall_clock_s']} s", flush=True)
    return summary
