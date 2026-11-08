"""Collect run summaries, compute the plan's metrics and decision rule, draw the figures."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Chart colours (reference data-viz palette). Roundness levels are ordered, so they use one
# blue ramp light -> dark; conditions are categorical and take the fixed slot order.
LEVEL_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_runs(results: Path) -> list[dict]:
    runs = []
    for f in sorted(results.glob("*/summary.json")):
        if f.parent.name.startswith("_"):
            continue
        d = json.load(open(f))
        rec = {**d["meta"], **d["summary"], "dir": str(f.parent)}
        rec.setdefault("run_id", f.parent.name)
        rec.setdefault("phase", rec["run_id"].split("_")[0])
        runs.append(rec)
    return runs


def write_runs_table(runs, path):
    keys = sorted({k for r in runs for k in r if not isinstance(r[k], (list, dict))})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k, "") for k in keys})


def _mean_sd(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, int(len(v))


def hopper_flow_table(runs, levels, phase, gravity=None, bond=None):
    """Per level: mean Q (kg/s), sd, n for flow runs of one phase/condition."""
    out = {}
    for lv in levels:
        sel = [r for r in runs if r.get("kind") == "hopper" and r.get("phase") == phase and r["level"] == lv
               and r.get("mode") == "flow" and (gravity is None or r["gravity"] == gravity)
               and (bond is None or abs(r["bond"] - bond) < 1e-9)]
        out[lv] = _mean_sd([r["Q_kg_s"] for r in sel])
    return out


def gain_fraction(qtab, levels):
    q0, q5 = qtab[levels[0]][0], qtab[levels[-1]][0]
    if not (np.isfinite(q0) and np.isfinite(q5)) or q5 == q0:
        return {lv: float("nan") for lv in levels}
    return {lv: (qtab[lv][0] - q0) / (q5 - q0) for lv in levels}


def chute_table(runs, levels, angles):
    """flows[level][angle] = (fraction of seeds that flow, mean exit/feed, mean station speed)."""
    tab = defaultdict(dict)
    for lv in levels:
        for a in angles:
            sel = [r for r in runs if r.get("kind") == "chute" and r["level"] == lv and abs(r["angle_deg"] - a) < 1e-9
                   and r.get("gravity") == "lunar"]
            if sel:
                tab[lv][a] = (float(np.mean([r["flows"] for r in sel])),
                              _mean_sd([r["exit_over_feed"] for r in sel])[0],
                              _mean_sd([r["station_speed_m_s"] for r in sel])[0], len(sel))
    return tab


def theta_flow(ctab, levels):
    """Lowest angle at which every seed flows, and every steeper tested angle flows too."""
    out = {}
    for lv in levels:
        angs = sorted(ctab.get(lv, {}))
        val = float("nan")
        for i, a in enumerate(angs):
            if all(ctab[lv][b][0] == 1.0 for b in angs[i:]):
                val = a
                break
        out[lv] = val
    return out


def clog_table(runs, levels):
    """clog[level][outlet] = (clog events per kg discharged, mean avalanche kg, all runs clog-free, n runs)."""
    tab = defaultdict(dict)
    for r in runs:
        if r.get("kind") != "hopper" or r.get("gravity") != "lunar" or r.get("bond", 0) != 0:
            continue
        if r.get("phase") != "C":  # equal-length clog runs only (Phase C includes 6d)
            continue
        tab[r["level"]].setdefault(r["outlet_mult"], []).append(r)
    out = defaultdict(dict)
    for lv in levels:
        for D, rs in tab.get(lv, {}).items():
            av = [a for r in rs for a in r.get("avalanche_kg", [])]
            kg = sum(r["discharged_kg"] for r in rs)
            out[lv][D] = (sum(r["n_clogs"] for r in rs) / kg if kg > 0 else float("nan"),
                          float(np.mean(av)) if av else float("nan"),
                          all(r["n_clogs"] == 0 for r in rs), len(rs))
    return out


def critical_outlet(ctab, levels):
    """Smallest tested outlet from which every larger tested outlet is clog-free too."""
    out = {}
    for lv in levels:
        Ds = sorted(ctab.get(lv, {}))
        val = float("nan")
        for i, D in enumerate(Ds):
            if all(ctab[lv][E][2] for E in Ds[i:]):
                val = D
                break
        out[lv] = val
    return out


def decide(cfg, levels, gain, tflow, clog):
    rule = cfg["decision_rule"]
    R = cfg["levels"]
    D5 = rule["clog_free_outlet_multiple"]
    star, rmin, notes = None, None, []
    for lv in levels:
        g = gain.get(lv, float("nan"))
        okA = np.isfinite(g) and g >= rule["gain_target"]
        c = clog.get(lv, {}).get(D5)
        okC = c is not None and c[2]
        tf = tflow.get(lv, float("nan"))
        okB = np.isfinite(tf) and tf <= rule["chute_design_angle_deg"]
        if star is None and okA and okC and okB:
            star = lv
        if np.isfinite(g) and g < rule["gain_floor"]:
            rmin = lv
        notes.append({"level": lv, "R": R[lv], "gain": g, "gain_ok": bool(okA),
                      "clog_free_at_5d": None if c is None else bool(c[2]), "theta_flow_deg": tf,
                      "chute_ok": bool(okB)})
    knee = next((lv for lv in levels if np.isfinite(gain.get(lv, np.nan)) and gain[lv] >= rule["gain_target"]), None)
    return {"R_star_level": star, "R_star": R.get(star) if star else None,
            "R_min_level": rmin, "R_min": R.get(rmin) if rmin else None,
            "suggested_knee_level": knee, "per_level": notes, "rule": rule}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_hopper(cfg, levels, conditions, path):
    """Q/Q(L5) vs roundness, one line per gravity/cohesion condition."""
    plt = _plt()
    R = cfg["levels"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    _style(ax)
    drawn = 0
    for k, (label, qtab) in enumerate(conditions):
        ref = qtab[levels[-1]][0]
        xs, ys, es = [], [], []
        for lv in levels:
            q, sd, n = qtab[lv]
            if n and np.isfinite(ref) and ref > 0:
                xs.append(R[lv]); ys.append(q / ref); es.append(sd / ref)
        if not xs:
            continue
        c = CATEGORICAL[k % len(CATEGORICAL)]
        ax.errorbar(xs, ys, yerr=es, color=c, lw=2, marker="o", ms=6, capsize=3, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        drawn += 1
    ax.set_xlabel("Wadell roundness R", color=INK2)
    ax.set_ylabel("Discharge rate relative to spheres, Q/Q(L5)", color=INK2)
    ax.set_title("Hopper discharge vs particle roundness (30° wedge, 6d outlet)", color=INK, fontsize=11,
                 loc="left")
    ax.set_xlim(0, 1.12)
    if drawn > 1:  # every line ends at 1.0 by construction, so the legend (not end labels) names them
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_chute(cfg, levels, ctab, tflow, path):
    plt = _plt()
    R = cfg["levels"]
    design = cfg["decision_rule"]["chute_design_angle_deg"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    _style(ax)
    # Every tested case as a small mark: filled = flowed in all seeds, open = did not.
    for lv in levels:
        for a, (frac, _, _, _) in ctab.get(lv, {}).items():
            ax.plot(R[lv], a, marker="s", ms=7, color=LEVEL_RAMP[2],
                    markerfacecolor=LEVEL_RAMP[2] if frac == 1.0 else SURFACE, markeredgewidth=1.2, zorder=3)
    xs = [R[lv] for lv in levels if np.isfinite(tflow.get(lv, np.nan))]
    ys = [tflow[lv] for lv in levels if np.isfinite(tflow.get(lv, np.nan))]
    if xs:
        ax.plot(xs, ys, color=CATEGORICAL[1], lw=2, zorder=4, label="Lowest flowing angle")
    ax.axhline(design, color=INK2, lw=1, ls="--")
    ax.annotate(f"design angle {design}°", (0.02, design), xytext=(0, 4), textcoords="offset points",
                color=INK2, fontsize=8)
    ax.set_xlabel("Wadell roundness R", color=INK2)
    ax.set_ylabel("Chute angle (°)", color=INK2)
    ax.set_title("Lowest chute angle that keeps up with the feed (filled = flowed)", color=INK, fontsize=11,
                 loc="left")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_clog(cfg, levels, clog, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURFACE)
    _style(ax)
    for k, lv in enumerate(levels):
        Ds = sorted(clog.get(lv, {}))
        if not Ds:
            continue
        # Clog-free cases have no finite avalanche; mark them along the top edge, nudged per level.
        ys = [clog[lv][D][1] * 1e3 if not clog[lv][D][2] else np.nan for D in Ds]
        c = LEVEL_RAMP[k % len(LEVEL_RAMP)]
        ax.plot(Ds, ys, color=c, lw=2, marker="o", ms=6, label=f"{lv} (R {cfg['levels'][lv]:.2f})")
        free = [D for D in Ds if clog[lv][D][2]]
        if free:
            dx = (k - 0.5 * (len(levels) - 1)) * 0.07
            ax.plot([D + dx for D in free], [1.0] * len(free), ls="none", marker="^", ms=7, color=c,
                    markerfacecolor=SURFACE, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Outlet width D/d", color=INK2)
    ax.set_ylabel("Mean mass between clogs (g)", color=INK2)
    ax.set_title("Clogging vs outlet size (△ on top edge = no clog in any run)", color=INK, fontsize=11,
                 loc="left", pad=14)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_velocity_maps(cfg, runs, levels_to_show, path):
    plt = _plt()
    picks = []
    for lv in levels_to_show:
        cand = [r for r in runs if r.get("phase") == "A" and r["level"] == lv and r.get("seed") == 1
                and (Path(r["dir"]) / "snapshot.csv").exists()]
        if cand:
            picks.append((lv, Path(cand[0]["dir"]) / "snapshot.csv"))
    if not picks:
        return False
    fig, axs = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 4.2), facecolor=SURFACE, squeeze=False)
    vmax = 0.0
    data = []
    for lv, f in picks:
        a = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
        sp = np.hypot(a[:, 3], a[:, 5]) if len(a) else np.zeros(0)
        data.append((lv, a, sp))
        if len(sp):
            vmax = max(vmax, np.percentile(sp, 98))
    for ax, (lv, a, sp) in zip(axs[0], data):
        ax.set_facecolor(SURFACE)
        if len(a):
            sc = ax.scatter(a[:, 0] * 1e3, a[:, 2] * 1e3, c=sp, s=2, cmap="Blues", vmin=0, vmax=max(vmax, 1e-6))
        ax.set_aspect("equal")
        ax.set_title(f"{lv}  R = {cfg['levels'][lv]:.2f}", color=INK, fontsize=10)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlabel("x (mm)", color=INK2, fontsize=9)
    axs[0][0].set_ylabel("z (mm)", color=INK2, fontsize=9)
    fig.colorbar(sc, ax=axs[0].tolist(), shrink=0.8, label="speed (m/s)")
    fig.suptitle(f"Hopper speed field at t = {cfg['hopper']['snapshot_time_s']:g} s (Phase A, seed 1)",
                 color=INK, fontsize=11)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True
