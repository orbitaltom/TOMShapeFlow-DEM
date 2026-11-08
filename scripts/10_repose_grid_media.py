"""Pictures, videos and plots for the repose shape grid (CPU, off-screen PyVista + bundled ffmpeg).

    python scripts/10_repose_grid_media.py gallery            # template gallery (roundness x aspect grid)
    python scripts/10_repose_grid_media.py heaps              # final heap per shape (seed 1) + 12-panel montage
    python scripts/10_repose_grid_media.py video R013_A08     # one shape's slump video (seed 1 frames)
    python scripts/10_repose_grid_media.py compare            # the two time-synced comparison videos
    python scripts/10_repose_grid_media.py plots              # contour + angle-vs-roundness plots, results table
    python scripts/10_repose_grid_media.py speedscale         # recompute the fixed video speed scale
Everything is written to results/repose_grid/media/ (plots and table to results/repose_grid/).
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shapeflow.config import load_config, results_dir  # noqa: E402
from shapeflow.repose import heap_angle  # noqa: E402

GRID = results_dir() / "repose_grid"
MEDIA = GRID / "media"
SHAPES_DIR = Path(__file__).resolve().parent.parent / "shapes" / "repose_grid"
TEMPL = SHAPES_DIR / "templates"
R_LEVELS = ("013", "045", "080", "100")
A_LEVELS = ("06", "08", "10")
SURF, INK, INK2, GRID_C = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FLOOR = "#77756f"            # frozen floor layer (stills and videos)
GRAIN = "#e2dccd"            # single neutral grain colour (stills)
FIT = "#eb6834"              # fitted slope line
W = 1920                     # frame width (px); labels at FONT
FONT = 22
X_VIEW = (-0.006, 0.210)     # m along the floor: same orthographic view and zoom for every shape
Z_BOTTOM = -0.004            # m, bottom edge of the view
H = 720
PX_PER_M = W / (X_VIEW[1] - X_VIEW[0])
SCALE_BAR = (0.160, 0.170, 0.060)   # x0, x1, z (m): 10 mm bar, upper right above the heap


def key(r, a):
    return f"R{r}_A{a}"


def label(sh):
    if not sh.startswith("R"):
        return f"{sh} (shipped calibration shape, R 0.13, b/a 0.77, c/a 0.55)"
    r, a = int(sh[1:4]) / 100, int(sh[6:8]) / 10
    return f"roundness {r:.2f}, aspect {a:.1f}"


def templates(sh):
    """Grid templates, or the shipped shapes/templates/ for other shapes (e.g. the L0 anchor)."""
    tdir = TEMPL
    if sh not in json.load(open(TEMPL / "templates_index.json")):
        tdir = Path(__file__).resolve().parent.parent / "shapes" / "templates"
    idx = json.load(open(tdir / "templates_index.json"))[sh]
    out = []
    for m in idx:
        d = np.genfromtxt(tdir / m["file"], delimiter=",", names=True)
        d = np.atleast_1d(d)
        out.append((np.c_[d["x"], d["y"], d["z"]], np.asarray(d["r"])))
    return out


def quat_to_mat(q):
    """DEME Python quaternions are (x, y, z, w), local-to-global."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def spheres(temps, tid, pos, quat):
    """Global sphere centres and radii of every grain."""
    R = quat_to_mat(quat.astype(float))
    C, Rad, owner = [], [], []
    for k, (loc, rad) in enumerate(temps):
        sel = np.where(tid == k)[0]
        if not len(sel):
            continue
        c = pos[sel, None, :] + np.einsum("gij,sj->gsi", R[sel], loc)
        C.append(c.reshape(-1, 3))
        Rad.append(np.tile(rad, len(sel)))
        owner.append(np.repeat(sel, len(rad)))
    return np.vstack(C), np.concatenate(Rad), np.concatenate(owner)


def run_dir(sh, seed=1, g="earth"):
    """Run folder; a dt 2.5e-6 fallback rerun (after a memory/DEME abort) replaces the dt 5e-6 run when present."""
    fb = GRID / "runs" / f"{sh}_{g}_dt2.5e-06_s{seed}"
    return fb if (fb / "summary.json").exists() else GRID / "runs" / f"{sh}_{g}_s{seed}"


def fit_line(sh, seed=1, g="earth"):
    rd = run_dir(sh, seed, g)
    s = json.load(open(rd / "summary.json"))["summary"]
    h = np.genfromtxt(rd / "heap.csv", delimiter=",", names=True)
    P = np.c_[h["x"], h["y"], h["z"]]
    ang, prof = heap_angle(P, 0.002, 0.25, z0=s["floor_top_m"])
    hmax = prof[:, 1].max(); pk = int(np.argmax(prof[:, 1])); fk = prof[pk:]
    sel = (fk[:, 1] > 0.15 * hmax) & (fk[:, 1] < 0.85 * hmax)
    if sel.sum() < 3:
        return s, ang, None, None, None, None   # no fit (heap_angle also returns NaN then)
    m, c = np.polyfit(fk[sel, 0], fk[sel, 1], 1)
    return s, ang, m, c, fk[sel, 0].min(), fk[sel, 0].max()


def speed_scale(recompute=False):
    """Fixed speed colour scale for all videos: 95th percentile of every free grain's peak speed during its slump,
    pooled over all seed-1 runs with frames, rounded up to 0.05 m/s. Stored in media/speed_scale.json."""
    f = MEDIA / "speed_scale.json"
    if f.exists() and not recompute:
        return json.load(open(f))["vmax_m_s"]
    peaks, used = [], []
    for sh in [key(r, a) for a in A_LEVELS for r in R_LEVELS] + ["L0"]:
        d = run_dir(sh)                      # the chosen seed-1 run (fallback rerun if any)
        fd = d / "frames"
        if not ((d / "summary.json").exists() and (fd / "meta.npz").exists()):
            continue
        free = np.load(fd / "meta.npz")["family"] == 0
        vmax = None
        for fr in sorted(fd.glob("frame_*.npz")):
            v = np.linalg.norm(np.load(fr)["vel"][free], axis=1)
            vmax = v if vmax is None else np.maximum(vmax, v)
        peaks.append(vmax)
        used.append(d.name)
    p95 = float(np.percentile(np.concatenate(peaks), 95))
    vmax = round(math.ceil(p95 / 0.05) * 0.05, 2)
    MEDIA.mkdir(parents=True, exist_ok=True)
    json.dump({"vmax_m_s": vmax, "p95_raw_m_s": p95, "runs": used,
               "definition": "95th percentile of per-grain peak speed during the slump, pooled over the listed "
                             "seed-1 runs, rounded up to 0.05 m/s"}, open(f, "w"), indent=1)
    return vmax


class SideView:
    """Off-screen orthographic side view through the slice (looking along +y). Same camera and zoom for every
    shape. mode 'speed': free grains coloured by speed (fixed scale, colour bar); mode 'neutral': one grain colour."""

    def __init__(self, mode="speed", vmax=None, size=(W, H)):
        import pyvista as pv
        pv.OFF_SCREEN = True
        self.pv, self.mode, self.vmax = pv, mode, vmax
        self.p = pv.Plotter(off_screen=True, window_size=size)
        self.p.set_background(SURF)
        self.p.enable_parallel_projection()
        half_h = 0.5 * size[1] / PX_PER_M * (W / size[0])
        xm, zm = 0.5 * (X_VIEW[0] + X_VIEW[1]), Z_BOTTOM + half_h
        self.p.camera_position = [(xm, -0.8, zm), (xm, 0.01, zm), (0, 0, 1)]
        self.p.camera.parallel_scale = half_h
        self.actors = []
        self._static()

    def _static(self):
        pv, p = self.pv, self.p
        x0, x1, z = SCALE_BAR
        p.add_mesh(pv.Line((x0, -0.004, z), (x1, -0.004, z)), color=INK, line_width=5)
        p.add_point_labels([((x0 + x1) / 2, -0.004, z + 0.0035)], ["10 mm"], show_points=False, shape=None,
                           font_size=FONT, text_color=INK, always_visible=True, justification_horizontal="center")
        p.add_point_labels([((x0 + x1) / 2, -0.004, z - 0.006)], ["d = 2 mm"], show_points=False, shape=None,
                           font_size=FONT, text_color=INK2, always_visible=True, justification_horizontal="center")

    def frame(self, C, Rad, owner_fam, speed, title, lines=()):
        pv, p = self.pv, self.p
        for a in self.actors:
            p.remove_actor(a)
        self.actors = []
        fl = owner_fam == 1
        if fl.any():
            f = pv.PolyData(C[fl]); f["r"] = Rad[fl]
            self.actors.append(p.add_mesh(f, style="points_gaussian", render_points_as_spheres=True, color=FLOOR))
            p.mapper.SetScaleArray("r"); p.mapper.SetScaleFactor(1.0)
        fr = ~fl
        c = pv.PolyData(C[fr]); c["r"] = Rad[fr]
        if self.mode == "speed":
            c["speed (m/s)"] = speed[fr]
            self.actors.append(p.add_mesh(
                c, style="points_gaussian", render_points_as_spheres=True, scalars="speed (m/s)", cmap="viridis",
                clim=[0, self.vmax], show_scalar_bar=True,
                scalar_bar_args={"title": "speed (m/s)", "vertical": True, "position_x": 0.925, "position_y": 0.12,
                                 "width": 0.03, "height": 0.5, "fmt": "%.2f", "n_labels": 8, "color": INK,
                                 "title_font_size": FONT, "label_font_size": FONT - 2}))
        else:
            self.actors.append(p.add_mesh(c, style="points_gaussian", render_points_as_spheres=True, color=GRAIN,
                                          ambient=0.45, diffuse=0.7))
        p.mapper.SetScaleArray("r"); p.mapper.SetScaleFactor(1.0)
        for (x0, z0, x1, z1) in lines:
            self.actors.append(p.add_mesh(pv.Line((x0, -0.005, z0), (x1, -0.005, z1)), color=FIT, line_width=6))
        if title:
            self.actors.append(p.add_text(title, position="upper_left", font_size=FONT, color=INK))
        img = p.screenshot(return_img=True)
        if self.mode == "speed":
            p.remove_scalar_bar()
        return img


def frames_of(sh):
    fd = run_dir(sh) / "frames"
    meta = np.load(fd / "meta.npz")
    fs = sorted(fd.glob("frame_*.npz"))
    return meta, fs


def seed_stats(sh):
    """(seed-1 angle, mean, half range, n seeds) from the run summaries."""
    a = []
    for seed in (1, 2):
        f = run_dir(sh, seed) / "summary.json"
        if f.exists():
            a.append(json.load(open(f))["summary"]["angle_of_repose_deg"])
    return (a[0] if a else float("nan")), float(np.mean(a)) if a else float("nan"), \
        (0.5 * (max(a) - min(a)) if len(a) > 1 else None), len(a)


def heap_image(sh, view, title=True):
    temps = templates(sh)
    meta, fs = frames_of(sh)
    fr = np.load(fs[-1])
    C, Rad, own = spheres(temps, meta["template"], fr["pos"], fr["quat"])
    s, ang, m, c, x0, x1 = fit_line(sh)
    z0 = s["floor_top_m"]
    t = f"{label(sh)}   angle {ang:.1f}°   seed 1" if title else None
    img = view.frame(C, Rad, meta["family"][own], None, t,
                     lines=[(x0, m * x0 + c + z0, x1, m * x1 + c + z0)] if m is not None else [])
    return img, ang


def cmd_heaps():
    import imageio.v3 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MEDIA.mkdir(parents=True, exist_ok=True)
    view = SideView(mode="neutral")
    panels = {}
    for a in A_LEVELS:
        for r in R_LEVELS:
            sh = key(r, a)
            if not (run_dir(sh) / "frames_info.json").exists():   # finished runs only (a stopped run keeps its frames)
                continue
            img, ang = heap_image(sh, view, title=True)
            iio.imwrite(MEDIA / f"heap_{sh}.png", img)
            panels[sh], _ = heap_image(sh, view, title=False)
            print(f"heap {sh}: {ang:.2f} deg")
    if (run_dir("L0") / "frames_info.json").exists():      # calibration anchor: own still, not part of the grid montage
        img, ang = heap_image("L0", view, title=True)
        iio.imwrite(MEDIA / "heap_L0.png", img)
        print(f"heap L0: {ang:.2f} deg")
    nr, nc = len(A_LEVELS), len(R_LEVELS)
    # Fixed layout in inches so every panel has the image's own aspect and headers line up.
    pw = 6.0
    ph = pw * H / W
    gap, lab, left, top_t, top_h = 0.2, 0.6, 0.7, 0.7, 0.45
    fw = left + nc * (pw + gap)
    fh = top_t + top_h + nr * (ph + lab + gap)
    fig = plt.figure(figsize=(fw, fh), facecolor=SURF)
    any_two = False
    for i, a in enumerate(A_LEVELS[::-1]):
        y0 = fh - top_t - top_h - (i + 1) * (ph + lab + gap) + lab
        fig.text(left / 2 / fw, (y0 + ph / 2) / fh, f"aspect {int(a) / 10:.1f}", rotation=90, ha="center",
                 va="center", fontsize=16, color=INK)
        for j, r in enumerate(R_LEVELS):
            x0 = left + j * (pw + gap)
            if i == 0:
                fig.text((x0 + pw / 2) / fw, (fh - top_t - top_h / 2) / fh, f"roundness {int(r) / 100:.2f}",
                         ha="center", va="center", fontsize=16, color=INK)
            ax = fig.add_axes((x0 / fw, y0 / fh, pw / fw, ph / fh))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            sh = key(r, a)
            if sh in panels:
                ax.imshow(panels[sh], aspect="auto")
                a1, mean, half, n = seed_stats(sh)
                if n > 1:
                    any_two = True
                    txt = f"{mean:.1f} \u00b1 {half:.1f}\u00b0"
                else:
                    txt = f"{a1:.1f}\u00b0 (1 seed)"
                fig.text((x0 + pw / 2) / fw, (y0 - lab / 2) / fh, txt, ha="center", va="center", fontsize=20, color=INK)
            else:
                ax.set_facecolor("#f1f0ec")
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", fontsize=14,
                        color=INK2)
    note = ("angle = mean of seeds 1-2 \u00b1 half the seed range" if any_two else "angle = seed 1")
    fig.text(0.5, 1 - (top_t / 2) / fh, "Final heaps (seed 1 shown), side view through the 20 mm slice; floor layer "
             f"grey, fit line orange. {note}.", ha="center", va="center", fontsize=14, color=INK)
    fig.savefig(MEDIA / "heaps_montage.png", dpi=100, facecolor=SURF)
    print(f"montage -> {MEDIA / 'heaps_montage.png'}")


def video_frames(sh, view, every=1):
    temps = templates(sh)
    meta, fs = frames_of(sh)
    s, ang, m, c, x0, x1 = fit_line(sh)
    for f in fs[::every]:
        fr = np.load(f)
        C, Rad, own = spheres(temps, meta["template"], fr["pos"], fr["quat"])
        speed = np.linalg.norm(fr["vel"].astype(float), axis=1)[own]
        title = (f"{label(sh)}   final angle {ang:.1f}°   seed 1\n"
                 f"t = {float(fr['t']):.2f} s (simulated)   0.5x speed")
        yield float(fr["t"]), view.frame(C, Rad, meta["family"][own], speed, title)


def write_mp4(path, frames, fps=25):
    import imageio_ffmpeg
    wr = None
    n = 0
    for img in frames:
        if wr is None:
            h, w = img.shape[:2]
            wr = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264", quality=7,
                                             macro_block_size=8, pix_fmt_out="yuv420p")
            wr.send(None)
        wr.send(np.ascontiguousarray(img[..., :3]).tobytes())
        n += 1
    if wr is not None:
        wr.close()
    return n


def cmd_video(sh):
    MEDIA.mkdir(parents=True, exist_ok=True)
    view = SideView(mode="speed", vmax=speed_scale())
    out = MEDIA / f"slump_{sh}.mp4"
    n = write_mp4(out, (img for _, img in video_frames(sh, view)))   # 0.02 s frames at 25 fps = 0.5x speed
    print(f"video {sh}: {n} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


def cmd_compare():
    MEDIA.mkdir(parents=True, exist_ok=True)
    sets = {"compare_roundness_aspect0.8.mp4": [key(r, "08") for r in R_LEVELS],
            "compare_aspect_roundness0.80.mp4": [key("080", a) for a in A_LEVELS]}
    for name, shapes in sets.items():
        shapes = [s for s in shapes if (run_dir(s) / "frames_info.json").exists()]
        if not shapes:
            continue
        views = [SideView(mode="speed", vmax=speed_scale(), size=(W, H)) for _ in shapes]   # full panel height: labels clear the heap
        gens = [list(video_frames(s, v)) for s, v in zip(shapes, views)]   # (t, img) per shape
        n = max(len(g) for g in gens)
        stacked = (np.vstack([g[min(i, len(g) - 1)][1] for g in gens]) for i in range(n))
        k = write_mp4(MEDIA / name, stacked)
        print(f"{name}: {len(shapes)} panels, {k} frames")


def cmd_gallery():
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pv.OFF_SCREEN = True
    MEDIA.mkdir(parents=True, exist_ok=True)
    summ = {r["shape"]: r for r in csv.DictReader(open(SHAPES_DIR / "shapes_summary.csv"))}
    fig, axs = plt.subplots(len(A_LEVELS), len(R_LEVELS), figsize=(3.2 * len(R_LEVELS), 3.3 * len(A_LEVELS)),
                            facecolor=SURF)
    for i, a in enumerate(A_LEVELS[::-1]):
        for j, r in enumerate(R_LEVELS):
            sh = key(r, a)
            ax = axs[i, j]
            ax.axis("off")
            if sh not in summ:
                continue
            loc, rad = templates(sh)[0]
            p = pv.Plotter(off_screen=True, window_size=(600, 600))
            p.set_background(SURF)
            mesh = pv.MultiBlock([pv.Sphere(radius=float(rr), center=tuple(cc), theta_resolution=32, phi_resolution=32)
                                  for cc, rr in zip(loc, rad)]).combine()
            p.add_mesh(mesh, color="#9bb7d4", smooth_shading=True, specular=0.3)
            p.camera_position = [(0.0042, -0.0058, 0.0035), (0, 0, 0), (0, 0, 1)]
            p.camera.parallel_projection = True
            p.camera.parallel_scale = 0.0021
            img = p.screenshot(return_img=True)
            p.close()
            ax.imshow(img)
            m = summ[sh]
            ax.set_title(f"{sh}: R {float(m['roundness_measured_mean']):.2f}, b/a {float(m['b_a_mean']):.2f}, "
                         f"c/a {float(m['c_a_mean']):.2f}\n{float(m['spheres_mean']):.0f} spheres, "
                         f"sphericity {float(m['sphericity_mean']):.2f}", fontsize=8.5, color=INK)
    fig.suptitle("Template 0 of each shape, same scale (columns: roundness target 0.13 / 0.45 / 0.80 / 1.0; "
                 "rows: aspect target 1.0 / 0.8 / 0.6)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(SHAPES_DIR / "template_gallery.png", dpi=120, facecolor=SURF)
    print(f"gallery -> {SHAPES_DIR / 'template_gallery.png'}")


def cmd_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    allrows = [r for r in csv.DictReader(open(GRID / "grid_results.csv")) if r["gravity"] == "earth"]
    rows = [r for r in allrows if r["shape"].startswith("R")]
    anchor = [r for r in allrows if r["shape"] == "L0"]
    S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
    crr = load_config()["particles"]["Crr"]
    plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "font.size": 10})
    x = np.array([float(r["roundness_measured"]) for r in rows])
    y = np.array([0.5 * (float(r["b_a"]) + float(r["c_a"])) for r in rows])
    z = np.array([float(r["mean"]) for r in rows])
    # existing calibration-test levels (same test, mu 0.6, Crr 0.25, seed 1)
    # Optional overlay: the earlier calibration-test levels L0-L5 (mu 0.6, Crr 0.25, seed 1). Their files
    # (shapes/levels_summary.csv, results/repose/aor_vs_R.csv) are not part of this release; without them the overlay
    # is skipped and the L0 anchor star uses the measured L0 values below.
    lv_f = Path(__file__).resolve().parent.parent / "shapes" / "levels_summary.csv"
    ao_f = results_dir() / "repose" / "aor_vs_R.csv"
    lv = list(csv.DictReader(open(lv_f))) if lv_f.exists() else []
    ang_old = ({r["level"]: float(r["angle_deg"]) for r in csv.DictReader(open(ao_f)) if float(r["Crr"]) == 0.25}
               if ao_f.exists() else {})
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=150, facecolor=SURF)
    ax.set_facecolor(SURF)
    if len(rows) >= 4:
        tri = Triangulation(x, y)
        cs = ax.tricontourf(tri, z, levels=12, cmap="viridis")
        ax.tricontour(tri, z, levels=12, colors=SURF, linewidths=0.6)
        cb = fig.colorbar(cs, ax=ax)
        cb.set_label("angle of repose, mean of 2 seeds (deg)")
    ax.scatter(x, y, s=36, color=SURF, edgecolor=INK, linewidth=1.2, zorder=3, label="grid shapes")
    ns = [int(r.get("n_seeds") or 2) for r in rows]
    for xi, yi, zi, n in zip(x, y, z, ns):
        ax.annotate(f"{zi:.1f}" + (" (1 seed)" if n == 1 else ""), (xi, yi), xytext=(5, 4), textcoords="offset points",
                    fontsize=8, color=INK)
    ox = [float(r["corner_r_over_r_in_mean"]) for r in lv if r["level"] in ang_old]
    oy = [float(r["aspect_b_a_mean"]) for r in lv if r["level"] in ang_old]
    if ox:
        ax.scatter(ox, oy, marker="s", s=40, color=S2, edgecolor=SURF, linewidth=1.5, zorder=4,
                   label="existing L0-L5, Crr 0.25 (b/a shown; their c/a is 0.55-0.69)")
    for r in lv:
        if r["level"] in ang_old:
            ax.annotate(f"{r['level']} {ang_old[r['level']]:.1f}", (float(r["corner_r_over_r_in_mean"]),
                        float(r["aspect_b_a_mean"])), xytext=(5, -11), textcoords="offset points", fontsize=8, color=INK2)
    l0 = ([r for r in lv if r["level"] == "L0"] or
          [{"corner_r_over_r_in_mean": "0.132", "aspect_b_a_mean": "0.765"}])[0]   # measured L0 roundness, b/a
    star = (f"L0 calibration anchor at Crr {crr:g}: {float(anchor[0]['mean']):.1f} deg" if anchor else
            "L0 calibration point (Crr 0.25: 40.35 deg seed 1; 41.54 deg 3-seed mean)")
    ax.scatter([float(l0["corner_r_over_r_in_mean"])], [float(l0["aspect_b_a_mean"])], marker="*", s=220, color=S3,
               edgecolor=INK, linewidth=0.8, zorder=5, label=star)
    ax.set_xlabel("roundness, measured (mean corner r / r_in)")
    ax.set_ylabel("aspect ratio, measured ((b/a + c/a) / 2)")
    ax.set_title(f"Angle of repose over roundness and aspect ratio (Earth g, mu 0.6, Crr {crr:g})", loc="left",
                 fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(GRID / "aor_contours.png", facecolor=SURF)
    # angle vs roundness, one line per aspect
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150, facecolor=SURF)
    ax.set_facecolor(SURF)
    for col, a in zip((S1, S2, S3), ("0.6", "0.8", "1.0")):
        rr = sorted([r for r in rows if float(r["aspect_target"]) == float(a)], key=lambda r: float(r["roundness_measured"]))
        if not rr:
            continue
        xs = [float(r["roundness_measured"]) for r in rr]
        ys = [float(r["mean"]) for r in rr]
        lo = [float(r["mean"]) - 0.5 * float(r["spread"] or 0) for r in rr]
        hi = [float(r["mean"]) + 0.5 * float(r["spread"] or 0) for r in rr]
        ax.plot(xs, ys, lw=2, color=col, label=f"aspect {a}")
        ax.vlines(xs, lo, hi, color=col, lw=2)
        ax.scatter(xs, ys, s=60, color=col, edgecolor=SURF, linewidth=2, zorder=3)
        for xx, yy, r in zip(xs, ys, rr):
            if int(r.get("n_seeds") or 2) == 1:
                ax.annotate("1 seed", (xx, yy), xytext=(6, -12), textcoords="offset points", fontsize=8, color=INK2)
    Rs = np.linspace(0, 1.05, 50)
    ax.plot(Rs, 42 - 17 * Rs, ls=(0, (5, 4)), lw=2, color=INK2, label="42 - 17R (Cho et al. 2006)")
    ax.set_xlabel("roundness, measured")
    ax.set_ylabel("angle of repose (deg)")
    ax.set_title("Angle of repose vs roundness, by aspect ratio (mean of 2 seeds; bars = seed range)", loc="left",
                 fontsize=10.5)
    ax.grid(True, color=GRID_C, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(GRID / "aor_vs_roundness_by_aspect.png", facecolor=SURF)
    print("plots written")


if __name__ == "__main__":
    c = sys.argv[1]
    {"gallery": cmd_gallery, "heaps": cmd_heaps, "compare": cmd_compare, "plots": cmd_plots,
     "speedscale": lambda: print("speed scale (m/s):", speed_scale(recompute=True))}.get(
        c, lambda: cmd_video(sys.argv[2]))()
