"""Pure-Python fallback ray engine. Same API surface as the native
mesh_baker_core.Scene, built on mathutils.bvhtree. Slow but always works —
used when the compiled core isn't available for the platform."""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .maps import cosine_cone_samples, onb_from_normals, barycentric


class PyScene:
    def __init__(self, verts, tris, tri_mesh):
        self.verts = verts
        self.tris = tris.astype(np.int64)
        self.tri_mesh = tri_mesh
        vlist = [tuple(v) for v in verts.tolist()]
        plist = [tuple(t) for t in tris.tolist()]
        self.bvh = BVHTree.FromPolygons(vlist, plist, all_triangles=True)

    # ------------------------------------------------------------ helpers
    def _first_allowed(self, o, d, tmax, eps, lmesh, allow):
        """First allowed hit along d. Returns (tri, t, hitco) or None."""
        base = 0.0
        ov = Vector(o)
        dv = Vector(d)
        for _ in range(24):
            remaining = tmax - base
            if remaining <= eps:
                return None
            loc, _nrm, idx, dist = self.bvh.ray_cast(ov + dv * base, dv, remaining)
            if idx is None:
                return None
            t = base + dist
            tm = int(self.tri_mesh[idx])
            if allow is None or allow[lmesh, tm]:
                return idx, t, np.array(loc, np.float32)
            base = t + max(eps, t * 1e-5)
        return None

    def _tri_pts(self, tri):
        t = self.tris[tri]
        return self.verts[t[0]], self.verts[t[1]], self.verts[t[2]]

    # ------------------------------------------------------------ project
    def project(self, origins, dirs, frontal, rear, mode, cage_len,
                tex_mesh, allow, eps):
        n = len(origins)
        r_tri = np.full(n, -1, np.int32)
        r_t = np.zeros(n, np.float32)
        r_u = np.zeros(n, np.float32)
        r_v = np.zeros(n, np.float32)
        r_pos = origins.copy()
        has_allow = allow is not None and allow.size > 0
        A = allow if has_allow else None
        for i in range(n):
            o = origins[i]
            d = dirs[i]
            lm = int(tex_mesh[i]) if tex_mesh is not None and len(tex_mesh) else 0
            if mode == 1:  # cage
                ln = float(cage_len[i])
                hit = self._first_allowed(o, d, ln + rear, eps, lm, A)
                if hit is None:
                    continue
                tri, t, co = hit
                signed = ln - t
            else:
                hf = self._first_allowed(o, d, frontal, eps, lm, A) if frontal > 0 else None
                hb = self._first_allowed(o, -d, rear, eps, lm, A) if rear > 0 else None
                if hf is not None and (hb is None or hf[1] <= hb[1]):
                    tri, t, co = hf
                    signed = t
                elif hb is not None:
                    tri, t, co = hb
                    signed = -t
                else:
                    continue
            r_tri[i] = tri
            r_t[i] = signed
            a, b, c = self._tri_pts(tri)
            u, v = barycentric(co[None], a[None], b[None], c[None])
            r_u[i] = u[0]
            r_v[i] = v[0]
            r_pos[i] = co
        return r_tri, r_t, r_u, r_v, r_pos

    # ---------------------------------------------------------- occlusion
    def occlusion(self, points, normals, nrays, maxdist, spread_deg, atten,
                  ignore_backface, self_mode, pt_mesh, seed, eps):
        n = len(points)
        out = np.ones(n, np.float32)
        tfar = maxdist if maxdist > 0 else 1e30
        md = maxdist if maxdist > 0 else 0.0
        t_axis, b_axis = onb_from_normals(normals)
        rng = np.random.default_rng(seed)
        local = cosine_cone_samples(nrays, spread_deg, rng)  # shared sample set
        # per-point random azimuth rotation
        ang = rng.random(n).astype(np.float32) * 2.0 * np.pi
        ca, sa = np.cos(ang), np.sin(ang)
        for i in range(n):
            lx = local[:, 0] * ca[i] - local[:, 1] * sa[i]
            ly = local[:, 0] * sa[i] + local[:, 1] * ca[i]
            dirs = (lx[:, None] * t_axis[i] + ly[:, None] * b_axis[i]
                    + local[:, 2][:, None] * normals[i])
            o = Vector(points[i] + normals[i] * eps)
            pm = int(pt_mesh[i]) if pt_mesh is not None and len(pt_mesh) else 0
            occ = 0.0
            for d in dirs:
                dv = Vector(d)
                base = 0.0
                for _ in range(12):
                    loc, ng, idx, dist = self.bvh.ray_cast(o + dv * base, dv, tfar - base)
                    if idx is None:
                        break
                    t = base + dist
                    tm = int(self.tri_mesh[idx])
                    counts = True
                    if ignore_backface and dv.dot(ng) > 0.0:
                        counts = False
                    if self_mode == 1 and tm != pm:
                        counts = False
                    if self_mode == 2 and tm == pm:
                        counts = False
                    if not counts:
                        base = t + max(eps, t * 1e-5)
                        if base >= tfar:
                            break
                        continue
                    w = 1.0
                    if md > 0.0:
                        x = max(0.0, 1.0 - t / md)
                        if atten == 1:
                            w = x
                        elif atten == 2:
                            w = x * x * (3.0 - 2.0 * x)
                    occ += w
                    break
            out[i] = 1.0 - occ / nrays
        return out

    # ---------------------------------------------------------- thickness
    def thickness(self, points, normals, nrays, maxdist, spread_deg, seed, eps):
        n = len(points)
        out = np.ones(n, np.float32)
        md = max(maxdist, 1e-8)
        inward = -normals
        t_axis, b_axis = onb_from_normals(inward)
        rng = np.random.default_rng(seed ^ 0x5EED)
        local = cosine_cone_samples(nrays, spread_deg, rng)
        ang = rng.random(n).astype(np.float32) * 2.0 * np.pi
        ca, sa = np.cos(ang), np.sin(ang)
        for i in range(n):
            lx = local[:, 0] * ca[i] - local[:, 1] * sa[i]
            ly = local[:, 0] * sa[i] + local[:, 1] * ca[i]
            dirs = (lx[:, None] * t_axis[i] + ly[:, None] * b_axis[i]
                    + local[:, 2][:, None] * inward[i])
            o = Vector(points[i] + inward[i] * eps)
            acc = 0.0
            for d in dirs:
                loc, _ng, idx, dist = self.bvh.ray_cast(o, Vector(d), md)
                acc += (dist if idx is not None else md) / md
            out[i] = acc / nrays
        return out
