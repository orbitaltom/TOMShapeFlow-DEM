"""Clump templates for the repose shape grid: roundness x aspect ratio (b/a = c/a).

Built on shapes_gen (rounding by morphological opening + greedy sphere cover) with two controls added:
- aspect: each random base polyhedron is stretched along its principal axes, rounded, covered and measured; the
  stretch is corrected until the measured b/a and c/a are both within +-ASPECT_TOL of the target (else the base
  is discarded and a new one drawn);
- roundness: the design roundness of a shape is scaled until the measured mean corner ratio (smallest hull-sphere
  radius / inscribed radius, as in levels_summary.csv) is within +-ROUND_TOL of the target.
R = 1 with aspect < 1 is a capsule (spherocylinder, Wadell roundness 1 by construction); R = 1 at aspect 1 is a
sphere. Surface bumps (true surface outside the sphere union) are measured for every template.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

from . import shapes_gen as sg

GRID_R = (0.13, 0.45, 0.80, 1.0)
GRID_AR = (0.6, 0.8, 1.0)
ASPECT_TOL = 0.03
ROUND_TOL = 0.05
N_VOX = 144


def shape_key(R, ar):
    return f"R{int(round(R * 100)):03d}_A{int(round(ar * 10)):02d}"


def bump_tol(R):
    """Surface tolerance as a fraction of d_v: 3.5% for the sharpest shapes (GPU memory), 2% otherwise."""
    return 0.035 if R < 0.2 else 0.02


def _hull(pts):
    hull = ConvexHull(pts)
    eq = hull.equations
    _, idx = np.unique(np.round(eq, 6), axis=0, return_index=True)
    eq = eq[np.sort(idx)]
    return eq[:, :3], eq[:, 3], pts[hull.vertices]


def _stretch(pts, tb, tc):
    """Rescale points along their principal axes so the extents are 1 : tb : tc (longest first)."""
    c = pts.mean(axis=0)
    q = pts - c
    w, V = np.linalg.eigh(q.T @ q)
    V = V[:, np.argsort(w)[::-1]]
    loc = q @ V
    ext = loc.max(axis=0) - loc.min(axis=0)
    loc = loc * (np.array([1.0, tb, tc]) * ext[0] / ext)
    return loc


def _capsule_voxels(ar):
    rc = 1.0
    hl = rc * (1.0 / ar - 1.0)
    lo = np.array([-hl - rc, -rc, -rc]) - 0.05
    h = (-2 * lo).max() / N_VOX
    axes, P = sg._grid(lo, -lo, h)
    x = np.clip(P[..., 0], -hl, hl)
    occ = (P[..., 0] - x) ** 2 + P[..., 1] ** 2 + P[..., 2] ** 2 <= rc * rc
    return occ, axes, h, rc


def _bumps(occ, axes, h, C, rad, d_v):
    surf = occ & ~ndimage.binary_erosion(occ)
    idx = np.argwhere(surf)
    S = np.stack([axes[k][idx[:, k]] for k in range(3)], axis=1)
    k = min(16, len(C))
    dd, ii = cKDTree(C).query(S, k=k)
    dd, ii = dd.reshape(len(S), -1), ii.reshape(len(S), -1)
    gap = np.maximum((dd - rad[ii]).min(axis=1) - 0.5 * h, 0.0)
    return float(100 * np.percentile(gap, 97) / d_v), float(100 * gap.max() / d_v)


def _cover(occ, axes, h, r_corner, tol_d):
    vol = occ.sum() * h ** 3
    d_v = (6 * vol / math.pi) ** (1 / 3)
    tol = max(h, tol_d * d_v)
    C, rad, frac = sg.greedy_sphere_cover(occ, axes, h, r_corner, max_spheres=600, coverage=0.97, tol=tol)
    b97, bmax = _bumps(occ, axes, h, C, rad, d_v)
    return C, rad, float(frac), b97, bmax


def polyhedral_template(key, R_target, R_design, ar, k, d_target, density, seed, max_bases=20, max_iter=6):
    """One accepted template (aspect within tolerance) or None. Returns (Template, extras)."""
    rng = np.random.default_rng([seed, k, int(R_target * 1000), int(ar * 1000)])
    tol_d = bump_tol(R_target)
    for base_try in range(max_bases):
        A0, b0, pts = sg.random_base_polyhedron(rng, axes=(1.0, ar, ar))
        tb = tc = ar
        for it in range(max_iter):
            A, b, verts = _hull(_stretch(pts, tb, tc))
            occ, axes_, h, r_in, r = sg.rounded_voxels(A, b, verts, R_design, n_vox=N_VOX)
            C, rad, frac, b97, bmax = _cover(occ, axes_, h, r, tol_d)
            t = sg._finalize(key, R_target, k, C, rad, d_target, density)
            db, dc = t.aspect_b_a - ar, t.aspect_c_a - ar
            if abs(db) <= ASPECT_TOL and abs(dc) <= ASPECT_TOL:
                return t, {"base_try": base_try, "iter": it, "cover": frac, "bump97_pct_d": b97,
                           "bumpmax_pct_d": bmax, "tol_pct_d": 100 * tol_d, "R_design": R_design}
            # Correct the base stretch; allow factors above 1 so equant targets (ar = 1) can be reached.
            tb = float(np.clip(tb * ar / max(t.aspect_b_a, 1e-3), 0.4, 1.3))
            tc = float(np.clip(tc * ar / max(t.aspect_c_a, 1e-3), 0.4, 1.3))
    return None, None


def capsule_templates(key, ar, d_targets, density):
    occ, axes_, h, rc = _capsule_voxels(ar)
    C, rad, frac, b97, bmax = _cover(occ, axes_, h, rc, bump_tol(1.0))
    out = []
    for k, d_t in enumerate(d_targets):
        t = sg._finalize(key, 1.0, k, C, rad, d_t, density)
        out.append((t, {"cover": frac, "bump97_pct_d": b97, "bumpmax_pct_d": bmax, "tol_pct_d": 100 * bump_tol(1.0),
                        "R_design": 1.0, "base_try": 0, "iter": 0}))
    return out


def sphere_templates(key, d_targets, density):
    return [(sg.sphere_template(key, k, d_t, density), {"cover": 1.0, "bump97_pct_d": 0.0, "bumpmax_pct_d": 0.0,
                                                         "tol_pct_d": 0.0, "R_design": 1.0, "base_try": 0, "iter": 0})
            for k, d_t in enumerate(d_targets)]


def template_meta(t, extra):
    m = asdict(t)
    m.pop("radii_m")
    m.pop("positions_m")
    m.update(extra)
    return m
