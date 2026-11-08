#!/usr/bin/env python3
"""Design chart, shape strip and design table for the repose shape study.

Reads the final results and draws:
  design_chart.png  angle of repose over measured roundness R and aspect ratio A,
                    from the linear surface in design_fit.json, with the 12 measured
                    shapes, the shape-target zone and a Chang'e-5 reference point
  shape_strip.png   five grains from crushed/elongated to round/equant with their angles
  design_table.csv  angle predicted by the linear surface on a regular R x A grid
                    (valid inside the simulated range only)

Usage (from the repository root):
  python scripts/11_design_chart.py
  python scripts/11_design_chart.py --results results/repose_grid/grid_results.csv \
      --fit results/repose_grid/design_fit.json --gallery shapes/repose_grid/template_gallery.png \
      --out results/repose_grid/figures

Needs numpy, pandas, matplotlib, pillow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

# Single-hue sequential ramp (light = low angle = flows better).
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTE, ACCENT = "#141414", "#555555", "#B84E1C"

# Shape-target zone (measured grid corner): R >= 0.8 and A >= 0.8.
TARGET_R, TARGET_A = 0.8, 0.8

# Chang'e-5 millimetre-scale grains (Chen 2025, Mendeley Data dataset; see the preprint references):
# mean Wadell roundness 0.12, aspect ratio 0.76 (aspect definition may differ from A here).
CE5 = (0.12, 0.76)

# Strip: crushed and elongated -> round and equant.
STRIP = [
    ("R013_A06", "Angular,\nelongated"),
    ("R013_A08", "Angular"),
    ("R045_A08", "Partly\nrounded"),
    ("R080_A08", "Rounded"),
    ("R100_A10", "Round,\nequant"),
]

# template_gallery.png layout: 4 columns (roundness 0.13, 0.45, 0.80, 1.0) x 3 rows
# (aspect 1.0, 0.8, 0.6), drawn on a 1536 x 1188 canvas. Crop boxes are given on that
# canvas and scaled to the actual image size.
GAL_REF = (1536, 1188)
GAL_ROWS = {1.0: (140, 370), 0.8: (520, 750), 0.6: (895, 1125)}
GAL_COLS = {0.13: 0, 0.45: 1, 0.80: 2, 1.0: 3}


def shape_targets(name: str) -> tuple[float, float]:
    r = int(name[1:4]) / 100.0
    a = int(name[6:8]) / 10.0
    return (0.80 if r == 0.8 else r), a


def load(results: Path, fit: Path):
    df = pd.read_csv(results)
    earth = df[(df["gravity"] == "earth") & df["shape"].str.match(r"R\d{3}_A\d{2}")].copy()
    earth["A"] = (earth["b_a"] + earth["c_a"]) / 2.0
    lunar = df[df["gravity"] == "lunar"].copy()
    fj = json.loads(Path(fit).read_text())
    lin = fj["forms"]["linear"]
    c = lin["coefficients"]
    return earth, lunar, (c["c0"], c["c_R"], c["c_A"]), lin["rmse_loo"]


def gallery_tiles(path: Path):
    img = Image.open(path).convert("RGB")
    sx, sy = img.width / GAL_REF[0], img.height / GAL_REF[1]

    def tile(rt: float, at: float):
        col = GAL_COLS[rt]
        y0, y1 = GAL_ROWS[at]
        x0 = col * 384 + 60
        box = (int(x0 * sx), int(y0 * sy), int((x0 + 264) * sx), int(y1 * sy))
        return np.asarray(img.crop(box))

    return tile


def design_chart(earth, lunar, coef, loo, tile, out: Path):
    c0, cR, cA = coef

    def f(R, A):
        return c0 + cR * R + cA * A

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig = plt.figure(figsize=(11.5, 8.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.22, 0.30, 0.60, 0.55])
    xlo, xhi, ylo, yhi = 0.08, 1.06, 0.56, 1.03
    RR, AA = np.meshgrid(np.linspace(xlo, xhi, 240), np.linspace(ylo, yhi, 240))
    Z = f(RR, AA)
    lv = np.arange(np.floor(Z.min()), np.ceil(Z.max()) + 0.01, 1.0)
    cf = ax.contourf(RR, AA, Z, levels=lv, cmap=cmap)
    cs = ax.contour(RR, AA, Z, levels=lv, colors="white", linewidths=0.8, alpha=0.9)

    # Contour labels placed between the rows of measured shapes so they never sit on a dot.
    def r_at(level, a):
        return (level - c0 - cA * a) / cR

    spots = []
    for level, a in ((38, 0.70), (37, 0.70), (36, 0.70), (35, 0.86), (34, 0.86), (33, 0.86), (32, 1.02)):
        r = r_at(level, a)
        if xlo + 0.02 < r < xhi - 0.02 and level in lv:
            spots.append((r, a))
    ax.clabel(cs, manual=spots, fmt="%d°", fontsize=8, colors="white", inline=True)

    # Shape-target zone.
    ax.add_patch(Rectangle((TARGET_R, TARGET_A), xhi - TARGET_R, yhi - TARGET_A, fill=False,
                           ec=ACCENT, lw=2.2, zorder=4))
    inzone = earth[(earth["roundness_measured"] >= TARGET_R - 0.02) & (earth["A"] >= TARGET_A - 0.02)]
    zlo, zhi = inzone["mean"].min(), inzone["mean"].max()
    ax.text(TARGET_R + 0.015, 0.945, f"Target zone\nR ≥ {TARGET_R:.1f}, A ≥ {TARGET_A:.1f}\n"
            f"measured {zlo:.1f}–{zhi:.1f}°", va="top", ha="left", fontsize=8.5, color=ACCENT,
            weight="bold", linespacing=1.3, zorder=6,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9))

    # Measured shapes.
    halo = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
    for _, r in earth.iterrows():
        x, y, v = r["roundness_measured"], r["A"], r["mean"]
        ax.plot(x, y, "o", ms=7, mfc=INK, mec="white", mew=1.5, zorder=5)
        right_edge = x > 0.95
        ax.annotate(f"{v:.1f}°", (x, y), xytext=(-7 if right_edge else 7, 6), textcoords="offset points",
                    ha="right" if right_edge else "left", fontsize=8.5, weight="bold", color=INK,
                    bbox=halo, zorder=6)

    # Chang'e-5 reference.
    ax.plot(*CE5, "*", ms=15, mfc=ACCENT, mec="white", mew=1, zorder=6)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Roundness R (Wadell, measured)  →  rounder", fontsize=10, color=INK, labelpad=52)
    ax.set_ylabel("Aspect ratio A = (b/a + c/a) / 2  →  more equant", fontsize=10, color=INK, labelpad=108)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTE, labelsize=9)

    # Thumbnails: x axis = aspect 0.8 row; y axis = roundness 1.0 column (elongation only).
    for rt in (0.13, 0.45, 0.80, 1.0):
        xm = earth.loc[earth["shape"] == f"R{int(round(rt * 100)):03d}_A08", "roundness_measured"].iloc[0]
        ax.add_artist(AnnotationBbox(OffsetImage(tile(rt, 0.8), zoom=0.16), (xm, ylo), xybox=(0, -36),
                                     xycoords="data", boxcoords="offset points", frameon=False,
                                     annotation_clip=False))
    for at in (1.0, 0.8, 0.6):
        ax.add_artist(AnnotationBbox(OffsetImage(tile(1.0, at), zoom=0.16), (xlo, at), xybox=(-80, 0),
                                     xycoords="data", boxcoords="offset points", frameon=False,
                                     annotation_clip=False))

    cax = fig.add_axes([0.85, 0.30, 0.018, 0.55])
    cb = fig.colorbar(cf, cax=cax)
    cb.set_label("Angle of repose (°)  ·  lower flows more freely", color=INK, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTE, labelsize=8)

    fig.text(0.22, 0.945, "Angle of repose by grain shape (DEM, 2 mm grains)", fontsize=15,
             weight="bold", color=INK)
    fig.text(0.22, 0.912, f"Surface: angle ≈ {c0:.1f} − {abs(cR):.1f}·R − {abs(cA):.1f}·A   "
             f"(±{loo:.1f}°, leave-one-out over 12 shapes)", fontsize=9.5, color=MUTE)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=7, mfc=INK, mec="white", label="Simulated shape (mean of 2 seeds)"),
        Line2D([], [], ls="none", marker="*", ms=12, mfc=ACCENT, mec="white",
               label="Chang'e-5 mm grains, R 0.12 / aspect 0.76 (Chen 2025)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.215, 0.902), ncol=2, frameon=False,
               fontsize=8.5, handletextpad=0.3, columnspacing=1.6, labelcolor=MUTE)

    resid = (earth["mean"] - f(earth["roundness_measured"], earth["A"])).abs().max()
    notes = [
        "DEM slump test at Earth gravity: μ 0.6, rolling resistance 0.22, calibrated so the reference angular grain "
        "gives 39.3° (LHS-1 datasheet: 39.6°).",
        f"The surface is a linear fit to {len(earth)} simulated shapes; individual shapes sit up to {resid:.1f}° off it. "
        "Read it as a trend, inside the simulated range only.",
    ]
    if len(lunar):
        dl = np.ceil(lunar["lunar_minus_earth"].abs().max() * 10) / 10
        notes.append(f"Lunar-gravity runs ({len(lunar)} shapes) agreed with Earth within {dl:.1f}°; "
                     "no gravity effect resolved with 2 seeds.")
    notes.append("Chang'e-5 point for context only; its aspect-ratio definition may differ from A. "
                 "Simulated grains, not measured regolith.")
    fig.text(0.22, 0.025, "\n".join(notes), fontsize=7.8, color=MUTE, linespacing=1.55, va="bottom")
    fig.savefig(out / "design_chart.png", facecolor="white")
    plt.close(fig)


def shape_strip(earth, tile, out: Path):
    fig, axs = plt.subplots(1, len(STRIP), figsize=(11, 3.5), dpi=150)
    fig.patch.set_facecolor("white")
    for ax, (name, label) in zip(axs, STRIP):
        rt, at = shape_targets(name)
        ax.imshow(tile(rt, at))
        ax.axis("off")
        row = earth[earth["shape"] == name].iloc[0]
        ax.set_title(label, fontsize=10, color=MUTE)
        ax.text(0.5, -0.10, f"{row['mean']:.1f}°", transform=ax.transAxes, ha="center", va="top",
                fontsize=18, weight="bold", color=INK)
        ax.text(0.5, -0.36, f"R {row['roundness_measured']:.2f} · A {row['A']:.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTE)
    fig.text(0.5, 0.95, "Angle of repose as grains get rounder and more equant (DEM, 2 mm grains, "
             "mean of 2 seeds)", ha="center", fontsize=12, weight="bold", color=INK)
    fig.subplots_adjust(top=0.76, bottom=0.27, wspace=0.15)
    fig.savefig(out / "shape_strip.png", facecolor="white")
    plt.close(fig)


def design_table(coef, earth, out: Path):
    c0, cR, cA = coef
    rs = np.round(np.arange(0.1, 1.001, 0.1), 2)
    as_ = np.round(np.arange(0.6, 1.001, 0.05), 2)
    rows = [{"A": a, **{f"R={r:.1f}": round(c0 + cR * r + cA * a, 1) for r in rs}} for a in as_[::-1]]
    t = pd.DataFrame(rows)
    note = (f"# angle of repose (deg) = {c0:.3f} + ({cR:.3f})*R + ({cA:.3f})*A; "
            f"R range {earth['roundness_measured'].min():.2f}-1.00, A range {earth['A'].min():.2f}-1.00; "
            "valid inside the simulated range only\n")
    (out / "design_table.csv").write_text(note + t.to_csv(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/repose_grid/grid_results.csv")
    ap.add_argument("--fit", default="results/repose_grid/design_fit.json")
    ap.add_argument("--gallery", default="shapes/repose_grid/template_gallery.png")
    ap.add_argument("--out", default="results/repose_grid/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    earth, lunar, coef, loo = load(Path(a.results), Path(a.fit))
    tile = gallery_tiles(Path(a.gallery))
    design_chart(earth, lunar, coef, loo, tile, out)
    shape_strip(earth, tile, out)
    design_table(coef, earth, out)
    print(f"wrote {out/'design_chart.png'}, {out/'shape_strip.png'}, {out/'design_table.csv'}")


if __name__ == "__main__":
    main()
