"""Chute and hopper geometry: triangle meshes for DEME (metres) and STEP solids for CAD (millimetres).

DEME treats a triangle's right-hand-rule normal as its contact side, so every mesh
here is wound with normals pointing toward the grains.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

IN = 0.0254  # metres per inch


# --------------------------------------------------------------------------
# OBJ meshes for DEME
# --------------------------------------------------------------------------

def _grid_quads(p0, du, dv, nu, nv):
    """Vertices and quads for a planar patch p0 + u*du + v*dv, u in [0,1], v in [0,1]."""
    verts = []
    for i in range(nu + 1):
        for j in range(nv + 1):
            verts.append(np.asarray(p0) + (i / nu) * np.asarray(du) + (j / nv) * np.asarray(dv))
    quads = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            quads.append((a, a + nv + 1, a + nv + 2, a + 1))
    return verts, quads


class Mesh:
    def __init__(self):
        self.v: list = []
        self.f: list = []

    def add_patch(self, p0, du, dv, cell, inward):
        """Add a planar rectangular patch, triangulated, with normals facing `inward`."""
        nu = max(1, int(math.ceil(np.linalg.norm(du) / cell)))
        nv = max(1, int(math.ceil(np.linalg.norm(dv) / cell)))
        verts, quads = _grid_quads(p0, du, dv, nu, nv)
        off = len(self.v)
        self.v.extend(verts)
        n = np.cross(np.asarray(du, float), np.asarray(dv, float))
        flip = float(np.dot(n, inward)) < 0
        for a, b, c, d in quads:
            tris = [(a, b, c), (a, c, d)]
            for t in tris:
                t = (t[0], t[2], t[1]) if flip else t
                self.f.append(tuple(off + k for k in t))

    def normals(self):
        V = np.asarray(self.v)
        out = []
        for a, b, c in self.f:
            n = np.cross(V[b] - V[a], V[c] - V[a])
            out.append(n / np.linalg.norm(n))
        return np.asarray(out)

    def write_obj(self, path, comment=""):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        N = self.normals()
        with open(path, "w") as f:
            f.write(f"# {comment}\n# units: metres; normals face the grains\n")
            for p in self.v:
                f.write(f"v {p[0]:.7f} {p[1]:.7f} {p[2]:.7f}\n")
            for n in N:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for k, (a, b, c) in enumerate(self.f, start=1):
                f.write(f"f {a + 1}//{k} {b + 1}//{k} {c + 1}//{k}\n")
        return path


def chute_floor_mesh(length, width, cell=0.01, overhang=0.002):
    """Floor of the open chute: x in [0, length], y across, z = 0, normal +z.

    The floor runs `overhang` past each side wall so no gap opens at the corners.
    """
    m = Mesh()
    half = 0.5 * width + overhang
    m.add_patch([0.0, -half, 0.0], [length, 0, 0], [0, 2 * half, 0], cell, inward=[0, 0, 1])
    return m


def hopper_half_width(z, top_width, outlet, theta_deg):
    """Inside half-width of the wedge hopper at height z above the outlet."""
    t = math.tan(math.radians(theta_deg))
    return min(0.5 * outlet + z * t, 0.5 * top_width)


def hopper_slope_height(top_width, outlet, theta_deg):
    return 0.5 * (top_width - outlet) / math.tan(math.radians(theta_deg))


def hopper_wall_mesh(top_width, outlet, theta_deg, z_top, slice_depth, cell=0.01, overhang=0.002):
    """Both sloped walls and vertical bin walls of the wedge hopper.

    x across, y through the slice (0 .. slice_depth), z up, outlet at z = 0.
    """
    m = Mesh()
    hs = hopper_slope_height(top_width, outlet, theta_deg)
    y0, ydep = -overhang, slice_depth + 2 * overhang
    for side in (-1.0, 1.0):
        inward = [-side, 0.0, 0.0]
        a = [side * 0.5 * outlet, y0, 0.0]
        b = [side * 0.5 * top_width, y0, hs]
        c = [side * 0.5 * top_width, y0, z_top]
        m.add_patch(a, np.subtract(b, a), [0, ydep, 0], cell, inward=np.add(inward, [0, 0, 0.5]))
        m.add_patch(b, np.subtract(c, b), [0, ydep, 0], cell, inward=inward)
    return m


# --------------------------------------------------------------------------
# STEP solids for CAD (needs the optional OCP package)
# --------------------------------------------------------------------------

def _ocp():
    try:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakePrism  # noqa: F401
        from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace  # noqa: F401
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse  # noqa: F401
        from OCP.gp import gp_Pnt, gp_Vec  # noqa: F401
        from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs  # noqa: F401
        from OCP.Interface import Interface_Static  # noqa: F401
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain  # noqa: F401
        from OCP.TopoDS import TopoDS_Compound  # noqa: F401
        from OCP.BRep import BRep_Builder  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("STEP export needs the 'cadquery-ocp' package (pip install cadquery-ocp)") from exc
    return locals()


MM = 25.4  # STEP solids are built and written in millimetres (OCCT's native unit); inputs are inches.


def _box(o, x0, y0, z0, x1, y1, z1):
    return o["BRepPrimAPI_MakeBox"](o["gp_Pnt"](x0 * MM, y0 * MM, z0 * MM),
                                    o["gp_Pnt"](x1 * MM, y1 * MM, z1 * MM)).Shape()


def _prism_xz(o, pts_xz, y0, depth):
    """Extrude a closed polygon given in the x-z plane along +y."""
    poly = o["BRepBuilderAPI_MakePolygon"]()
    for x, z in pts_xz:
        poly.Add(o["gp_Pnt"](x * MM, y0 * MM, z * MM))
    poly.Close()
    face = o["BRepBuilderAPI_MakeFace"](poly.Wire()).Face()
    return o["BRepPrimAPI_MakePrism"](face, o["gp_Vec"](0, depth * MM, 0)).Shape()


def _fuse(o, shapes):
    s = shapes[0]
    for t in shapes[1:]:
        s = o["BRepAlgoAPI_Fuse"](s, t).Shape()
    u = o["ShapeUpgrade_UnifySameDomain"](s, True, True, True)
    u.Build()
    return u.Shape()


def _compound(o, shapes):
    b = o["BRep_Builder"]()
    c = o["TopoDS_Compound"]()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


def _write_step(o, shape, path):
    o["Interface_Static"].SetCVal_s("write.step.schema", "AP214IS")
    w = o["STEPControl_Writer"]()
    w.Transfer(shape, o["STEPControl_AsIs"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    status = w.Write(str(path))
    return str(status)


def chute_step(path, length_in=14.14, width_in=2.0, wall_in=3.0, t_in=0.125):
    """Open 6061-T6 chute: floor, two side walls and a back plate, fused into one solid.

    Dimensions are given in inches; the STEP file is written in millimetres.

    Inside floor surface at z = 0, inside width `width_in`, flow toward +x, open at x = length_in.
    """
    o = _ocp()
    hw = 0.5 * width_in
    floor = _box(o, 0, -hw - t_in, -t_in, length_in, hw + t_in, 0)
    left = _box(o, 0, -hw - t_in, 0, length_in, -hw, wall_in)
    right = _box(o, 0, hw, 0, length_in, hw + t_in, wall_in)
    back = _box(o, -t_in, -hw - t_in, -t_in, 0, hw + t_in, wall_in)
    return _write_step(o, _fuse(o, [floor, left, right, back]), path)


def hopper_step(path, top_width_in=8.0, outlet_in=0.4724, theta_deg=30.0, height_in=16.0, slice_in=0.7874,
                t_in=0.125):
    """Wedge hopper of the DEM study (inputs in inches, file in millimetres): two 6061 walls (sloped + vertical) and two clear side plates.

    x across, y through the slice, z up, slot outlet at z = 0. Front and back plates are separate bodies.
    """
    o = _ocp()
    hs = 0.5 * (top_width_in - outlet_in) / math.tan(math.radians(theta_deg))
    walls = []
    for side in (-1.0, 1.0):
        # Inner line: outlet edge -> top of slope -> top of bin. Offset outward by t for the plate.
        x_out, x_top = side * 0.5 * outlet_in, side * 0.5 * top_width_in
        nx = side * math.cos(math.radians(theta_deg))  # outward normal of the sloped wall (x part)
        nz = -math.sin(math.radians(theta_deg))
        slope = [(x_out, 0.0), (x_top, hs), (x_top + nx * t_in, hs + nz * t_in), (x_out + nx * t_in, nz * t_in)]
        vert = [(x_top, hs), (x_top, height_in), (x_top + side * t_in, height_in), (x_top + side * t_in, hs)]
        walls.append(_prism_xz(o, slope, 0.0, slice_in))
        walls.append(_prism_xz(o, vert, 0.0, slice_in))
    wall_solid = _fuse(o, walls)
    span = 0.5 * top_width_in + t_in
    front = _box(o, -span, -t_in, nz_min(theta_deg, t_in), span, 0.0, height_in)
    back = _box(o, -span, slice_in, nz_min(theta_deg, t_in), span, slice_in + t_in, height_in)
    return _write_step(o, _compound(o, [wall_solid, front, back]), path)


def nz_min(theta_deg, t_in):
    return -math.sin(math.radians(theta_deg)) * t_in
