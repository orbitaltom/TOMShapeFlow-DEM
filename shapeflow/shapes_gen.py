"""Generate DEME clump templates at controlled Wadell roundness levels.

Method (CPU only, no DEME needed):
1. Build a jagged base particle: the convex hull of random points on an
   ellipsoid whose axis ratios put its aspect ratio near the Chang'e-5 mean.
2. Round it to level R by morphological opening with a ball of radius
   r = R x (inscribed radius). For a convex polyhedron this rounds every corner
   and edge to radius r, so the Wadell roundness of the result is R by
   construction.
3. Voxelize the rounded shape and cover its surface with inscribed spheres
   (greedy set cover within a tolerance). Sharp levels need small spheres at
   the corners, so they end up with more spheres.
4. Measure the union of spheres (volume, centre of mass, inertia, sphericity),
   move it to its principal frame and scale it to the requested
   volume-equivalent diameter.

The same base shapes are reused for every level, so between levels only the
roundness changes. L5 (R = 1) is a plain sphere.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize, sparse
from scipy.spatial import ConvexHull, cKDTree


@dataclass
class Template:
    level: str
    roundness_design: float
    index: int
    d_v_m: float  # volume-equivalent diameter
    mass_kg: float
    moi_kg_m2: list  # principal moments
    radii_m: list
    positions_m: list  # sphere centres in the principal frame, CoM at origin
    n_spheres: int
    sphericity: float  # Wadell 3D: area of equal-volume sphere / actual area
    aspect_b_a: float  # intermediate / longest principal extent
    aspect_c_a: float  # shortest / longest principal extent
    corner_radius_ratio: float  # smallest hull-touching sphere radius / inscribed radius
    bounding_radius_m: float  # max distance from CoM to any sphere surface


# --------------------------------------------------------------------------
# Base shapes
# --------------------------------------------------------------------------

def random_base_polyhedron(rng: np.random.Generator, axes=(1.0, 0.80, 0.62), n_points=16):
    """Return (A, b) with the convex polyhedron {x : A x + b <= 0}, unit normals in A."""
    pts = rng.normal(size=(n_points, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= np.asarray(axes) * (1.0 + 0.12 * rng.normal(size=(n_points, 1)))
    hull = ConvexHull(pts)
    eq = hull.equations  # n.x + d <= 0 inside, with unit n
    # Merge near-duplicate facets (coplanar triangles) to keep the LP small.
    keys = np.round(eq, 6)
    _, idx = np.unique(keys, axis=0, return_index=True)
    eq = eq[np.sort(idx)]
    return eq[:, :3], eq[:, 3], pts


def inscribed_ball(A, b):
    """Chebyshev centre: maximise t s.t. A x + b + t <= 0 (|A_i| = 1)."""
    n = A.shape[0]
    c = np.zeros(4)
    c[3] = -1.0
    A_ub = np.hstack([A, np.ones((n, 1))])
    b_ub = -b
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    if not res.success:
        raise RuntimeError("inscribed ball LP failed")
    return res.x[:3], res.x[3]


# --------------------------------------------------------------------------
# Rounding and sphere cover
# --------------------------------------------------------------------------

def _grid(lo, hi, h):
    axes = [np.arange(lo[i] + 0.5 * h, hi[i], h) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return axes, np.stack([X, Y, Z], axis=-1)


def rounded_voxels(A, b, verts, R, n_vox=96):
    """Voxelize the opening of polyhedron {A x + b <= 0} by a ball of radius R * r_in."""
    centre, r_in = inscribed_ball(A, b)
    r = R * r_in
    lo = verts.min(axis=0) - 0.05
    hi = verts.max(axis=0) + 0.05
    h = (hi - lo).max() / n_vox
    axes, P = _grid(lo, hi, h)
    plane_dist = (P @ A.T + b).max(axis=-1)  # >0 outside
    if r <= 0:
        occ = plane_dist <= 0
    else:
        eroded = plane_dist <= -r
        # Distance from each voxel to the eroded body, then dilate by r.
        dist_out = ndimage.distance_transform_edt(~eroded, sampling=h)
        occ = dist_out <= r
    return occ, axes, h, r_in, r


def greedy_sphere_cover(occ, axes, h, r_corner, max_spheres=60, coverage=0.985,
                        tol=None, n_surface=4000, cand_stride=2, rng=None):
    """Cover the surface voxels of `occ` with inscribed spheres (greedy set cover)."""
    rng = rng or np.random.default_rng(0)
    din = ndimage.distance_transform_edt(occ, sampling=h)  # distance to outside, inside voxels
    surf = occ & ~ndimage.binary_erosion(occ)
    idx_s = np.argwhere(surf)
    if len(idx_s) > n_surface:
        idx_s = idx_s[rng.choice(len(idx_s), n_surface, replace=False)]
    S = np.stack([axes[k][idx_s[:, k]] for k in range(3)], axis=1)

    r_min = max(0.6 * r_corner, 0.75 * h)
    mask = occ & (din >= r_min)
    sl = tuple(slice(0, None, cand_stride) for _ in range(3))
    sub = np.zeros_like(mask)
    sub[sl] = mask[sl]
    # Always keep local maxima of the distance field (medial-axis-like points).
    locmax = (din == ndimage.maximum_filter(din, size=3)) & mask
    cand_mask = sub | locmax
    idx_c = np.argwhere(cand_mask)
    C = np.stack([axes[k][idx_c[:, k]] for k in range(3)], axis=1)
    rad = din[cand_mask] - 0.5 * h  # conservative inscribed radius
    keep = rad > 0.5 * h
    C, rad = C[keep], rad[keep]

    if tol is None:
        tol = h
    tree = cKDTree(S)
    lists = tree.query_ball_point(C, r=rad + tol)
    rows = np.repeat(np.arange(len(C)), [len(l) for l in lists])
    cols = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists]) if len(rows) else np.array([], int)
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(C), len(S)))

    uncovered = np.ones(len(S), dtype=np.float32)
    chosen = []
    target = (1.0 - coverage) * len(S)
    while uncovered.sum() > target and len(chosen) < max_spheres:
        score = M @ uncovered
        j = int(np.argmax(score))
        if score[j] <= 0:
            break
        chosen.append(j)
        covered = M.getrow(j).indices
        uncovered[covered] = 0.0
    frac = 1.0 - uncovered.sum() / len(S)
    return C[chosen], rad[chosen], frac


# --------------------------------------------------------------------------
# Mass properties of a union of spheres
# --------------------------------------------------------------------------

def union_properties(centres, radii, n_vox=128):
    lo = (centres - radii[:, None]).min(axis=0)
    hi = (centres + radii[:, None]).max(axis=0)
    h = (hi - lo).max() / n_vox
    axes, P = _grid(lo - h, hi + h, h)
    occ = np.zeros(P.shape[:3], dtype=bool)
    for c, r in zip(centres, radii):
        occ |= ((P - c) ** 2).sum(-1) <= r * r
    pts = P[occ]
    vol = occ.sum() * h ** 3
    com = pts.mean(axis=0)
    q = pts - com
    I = np.zeros((3, 3))
    I[0, 0] = (q[:, 1] ** 2 + q[:, 2] ** 2).sum()
    I[1, 1] = (q[:, 0] ** 2 + q[:, 2] ** 2).sum()
    I[2, 2] = (q[:, 0] ** 2 + q[:, 1] ** 2).sum()
    I[0, 1] = I[1, 0] = -(q[:, 0] * q[:, 1]).sum()
    I[0, 2] = I[2, 0] = -(q[:, 0] * q[:, 2]).sum()
    I[1, 2] = I[2, 1] = -(q[:, 1] * q[:, 2]).sum()
    I *= h ** 3  # per unit density
    # Surface area via marching cubes on the padded occupancy grid.
    from skimage import measure
    verts, faces, _, _ = measure.marching_cubes(np.pad(occ, 1).astype(np.float32), 0.5, spacing=(h, h, h))
    area = measure.mesh_surface_area(verts, faces)
    din = ndimage.distance_transform_edt(occ, sampling=h)
    r_in = din.max()
    return vol, com, I, area, r_in


def corner_radius_ratio(centres, radii, r_in, n_dir=400, rng=None):
    """Smallest radius among spheres that form the convex hull of the clump, over r_in."""
    rng = rng or np.random.default_rng(1)
    d = rng.normal(size=(n_dir, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    # Support point in each direction belongs to the sphere maximising c.d + r.
    support = centres @ d.T + radii[:, None]
    owner = np.argmax(support, axis=0)
    hull_spheres = np.unique(owner)
    return float(radii[hull_spheres].min() / r_in)


# --------------------------------------------------------------------------
# Template builders
# --------------------------------------------------------------------------

def _finalize(level, R, k, centres, radii, d_target, density):
    vol, com, I, area, r_in = union_properties(centres, radii)
    w, V = np.linalg.eigh(I)  # principal moments (per unit density)
    local = (centres - com) @ V  # coordinates in the principal frame
    d_v = (6.0 * vol / math.pi) ** (1.0 / 3.0)
    s = d_target / d_v
    local *= s
    radii_s = radii * s
    vol_s = vol * s ** 3
    mass = density * vol_s
    moi = (w * s ** 5 * density).tolist()
    sphericity = (math.pi ** (1 / 3) * (6 * vol) ** (2 / 3)) / area
    ext = (local + radii_s[:, None]).max(axis=0) - (local - radii_s[:, None]).min(axis=0)
    ext_sorted = np.sort(ext)[::-1]
    crr = 1.0 if len(radii) == 1 else corner_radius_ratio(local, radii_s, r_in * s)
    bound = float((np.linalg.norm(local, axis=1) + radii_s).max())
    return Template(level=level, roundness_design=R, index=k, d_v_m=float(d_target), mass_kg=float(mass),
                    moi_kg_m2=[float(x) for x in moi], radii_m=[float(x) for x in radii_s],
                    positions_m=[[float(v) for v in p] for p in local], n_spheres=int(len(radii)),
                    sphericity=float(min(sphericity, 1.0)), aspect_b_a=float(ext_sorted[1] / ext_sorted[0]),
                    aspect_c_a=float(ext_sorted[2] / ext_sorted[0]), corner_radius_ratio=float(min(crr, 1.0)),
                    bounding_radius_m=bound)


def sphere_template(level, k, d_target, density):
    r = 0.5 * d_target
    mass = density * math.pi / 6.0 * d_target ** 3
    moi = 0.4 * mass * r * r
    return Template(level=level, roundness_design=1.0, index=k, d_v_m=float(d_target), mass_kg=float(mass),
                    moi_kg_m2=[moi, moi, moi], radii_m=[r], positions_m=[[0.0, 0.0, 0.0]], n_spheres=1,
                    sphericity=1.0, aspect_b_a=1.0, aspect_c_a=1.0, corner_radius_ratio=1.0, bounding_radius_m=r)


def build_level(level, R, bases, d_targets, density, max_spheres, tol_frac=0.03, coverage=0.97, log=print):
    out = []
    for k, ((A, b, verts), d_t) in enumerate(zip(bases, d_targets)):
        if R >= 0.999:
            out.append(sphere_template(level, k, d_t, density))
            continue
        occ, axes, h, r_in, r = rounded_voxels(A, b, verts, R)
        L = max(a[-1] - a[0] for a in axes)
        # Surface tolerance: how far the clump surface may sit inside the true surface.
        # Smaller = sharper corners and flatter faces, but more spheres per clump.
        tol = float(max(h, tol_frac * L))
        C, rad, frac = greedy_sphere_cover(occ, axes, h, r, max_spheres=max_spheres, coverage=coverage, tol=tol)
        t = _finalize(level, R, k, C, rad, d_t, density)
        log(f"  {level} t{k}: {t.n_spheres:2d} spheres, cover {frac:.3f}, sphericity {t.sphericity:.3f}, "
            f"b/a {t.aspect_b_a:.2f}, corner r/r_in {t.corner_radius_ratio:.2f}")
        out.append(t)
    return out


def generate_all(levels: dict, n_templates: int, d_min: float, d_max: float, density: float,
                 seed: int = 2026, max_spheres: int = 60, tol_frac: float = 0.03, coverage: float = 0.97,
                 log=print):
    rng = np.random.default_rng(seed)
    bases = []
    for _ in range(n_templates):
        A, b, pts = random_base_polyhedron(rng)
        bases.append((A, b, pts))
    d_targets = np.linspace(d_min, d_max, n_templates)
    rng.shuffle(d_targets)
    result = {}
    for level, R in levels.items():
        log(f"Level {level} (R = {R:.2f})")
        result[level] = build_level(level, R, bases, d_targets, density, max_spheres, tol_frac=tol_frac,
                                    coverage=coverage, log=log)
    return result


def write_templates(result: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for level, temps in result.items():
        index[level] = []
        for t in temps:
            fname = f"{level}_t{t.index}.csv"
            with open(out_dir / fname, "w") as f:
                f.write("x,y,z,r\n")
                for p, r in zip(t.positions_m, t.radii_m):
                    f.write(f"{p[0]:.9e},{p[1]:.9e},{p[2]:.9e},{r:.9e}\n")
            meta = asdict(t)
            meta.pop("radii_m")
            meta.pop("positions_m")
            meta["file"] = fname
            index[level].append(meta)
    with open(out_dir / "templates_index.json", "w") as f:
        json.dump(index, f, indent=1)
    return index


def preview(result: dict, png_path: Path, which: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    levels = list(result.keys())
    # One scale for every panel so sizes compare directly.
    lim = 1.15 * max(float((np.abs(np.asarray(result[lv][which].positions_m))[:, :2].max(axis=0).max()
                            + max(result[lv][which].radii_m)) * 1e3) for lv in levels)
    fig, axs = plt.subplots(1, len(levels), figsize=(2.3 * len(levels), 3.0))
    for ax, level in zip(np.atleast_1d(axs), levels):
        t = result[level][which]
        P = np.asarray(t.positions_m) * 1e3
        R = np.asarray(t.radii_m) * 1e3
        order = np.argsort(P[:, 2])  # draw back to front
        for i in order:
            ax.add_patch(Circle((P[i, 0], P[i, 1]), R[i], facecolor="#9bb7d4", edgecolor="#2b4a6b", lw=0.5,
                                alpha=0.9))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        noun = "sphere" if t.n_spheres == 1 else "spheres"
        ax.set_title(f"{level}  R = {t.roundness_design:.2f}\n{t.n_spheres} {noun}, sphericity {t.sphericity:.2f}",
                     fontsize=9)
    fig.suptitle(f"Template {which} at each roundness level, same scale (frame = {2 * lim:.1f} mm), "
                 "viewed along the shortest axis", fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
