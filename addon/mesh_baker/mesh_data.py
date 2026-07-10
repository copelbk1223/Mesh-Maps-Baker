"""Extraction of evaluated mesh data into flat NumPy arrays, and the
texel-space UV rasterizer that builds the g-buffer (like Substance's
first bake stage)."""
import numpy as np

from .maps import BakeError, normalize_rows


def object_arrays(obj, depsgraph, need_tangents, mat_name_offset=0, mesh_id=0):
    """Extract world-space triangle data from an evaluated object."""
    ob = obj.evaluated_get(depsgraph)
    me = ob.to_mesh()
    try:
        if need_tangents:
            if me.uv_layers.active is None:
                raise BakeError(f"'{obj.name}' has no UV map")
            me.calc_tangents()  # MikkTSpace, same standard as Substance/Marmoset
        me.calc_loop_triangles()
        # IMPORTANT: fetch the UV layer only AFTER calc_tangents/calc_loop_triangles.
        # Those calls reallocate the mesh's custom data, so any uv_layer
        # reference taken before them dangles and reads garbage (inf/NaN).
        uv_layer = me.uv_layers.active

        nv = len(me.vertices)
        nl = len(me.loops)
        nt = len(me.loop_triangles)
        if nt == 0:
            raise BakeError(f"'{obj.name}' has no faces")

        vco = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", vco)
        vco = vco.reshape(-1, 3)

        lvi = np.empty(nl, np.int32)
        me.loops.foreach_get("vertex_index", lvi)

        lno = np.empty(nl * 3, np.float32)
        try:  # Blender 4.1+
            me.corner_normals.foreach_get("vector", lno)
        except (AttributeError, RuntimeError):
            me.loops.foreach_get("normal", lno)
        lno = lno.reshape(-1, 3)

        ltan = lbs = luv = None
        if need_tangents:
            ltan = np.empty(nl * 3, np.float32)
            me.loops.foreach_get("tangent", ltan)
            ltan = ltan.reshape(-1, 3)
            lbs = np.empty(nl, np.float32)
            me.loops.foreach_get("bitangent_sign", lbs)
        if uv_layer is not None:
            luv = np.empty(nl * 2, np.float32)
            uv_layer.data.foreach_get("uv", luv)
            luv = luv.reshape(-1, 2)

        tl = np.empty(nt * 3, np.int32)
        me.loop_triangles.foreach_get("loops", tl)
        tl = tl.reshape(-1, 3)
        tp = np.empty(nt, np.int32)
        me.loop_triangles.foreach_get("polygon_index", tp)
        pmat = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("material_index", pmat)

        # world-space transform
        M = np.array(obj.matrix_world, dtype=np.float64)
        pos = (vco @ M[:3, :3].T + M[:3, 3]).astype(np.float32)
        nrm_m = np.linalg.inv(M[:3, :3]).T
        nrm = normalize_rows(lno @ nrm_m.T).astype(np.float32)
        if need_tangents:
            ltan = normalize_rows(ltan @ M[:3, :3].T).astype(np.float32)

        n_slots = max(1, len(obj.material_slots))
        mat_names = []
        for i in range(n_slots):
            m = obj.material_slots[i].material if i < len(obj.material_slots) else None
            mat_names.append(m.name if m else f"{obj.name}.slot{i}")

        return {
            "name": obj.name,
            "verts": pos,                              # (V, 3) world
            "tri_v": lvi[tl],                          # (T, 3) vertex indices
            "tri_nrm": nrm[tl],                        # (T, 3, 3) corner smooth normals
            "tri_uv": luv[tl] if luv is not None else None,
            "tri_tan": ltan[tl] if need_tangents else None,
            "tri_bs": lbs[tl] if need_tangents else None,
            "tri_mat": np.clip(pmat[tp], 0, n_slots - 1).astype(np.int32) + mat_name_offset,
            "tri_mesh": np.full(nt, mesh_id, np.int32),
            "mat_names": mat_names,
        }
    finally:
        ob.to_mesh_clear()


def merge_high(datas):
    """Concatenate multiple high-poly extractions into one indexed soup."""
    verts, tri_v, tri_nrm, tri_mat, tri_mesh, names, mesh_names = [], [], [], [], [], [], []
    off = 0
    for d in datas:
        verts.append(d["verts"])
        tri_v.append(d["tri_v"] + off)
        tri_nrm.append(d["tri_nrm"])
        tri_mat.append(d["tri_mat"])
        tri_mesh.append(d["tri_mesh"])
        names += d["mat_names"]
        mesh_names.append(d["name"])
        off += len(d["verts"])
    return {
        "verts": np.concatenate(verts),
        "tri_v": np.concatenate(tri_v).astype(np.uint32),
        "tri_nrm": np.concatenate(tri_nrm),
        "tri_mat": np.concatenate(tri_mat),
        "tri_mesh": np.concatenate(tri_mesh),
        "mat_names": names,
        "mesh_names": mesh_names,
    }


class GBuffer:
    """Per-texel interpolated low-poly surface data."""

    def __init__(self, res):
        self.res = res
        self.mask = np.zeros((res, res), bool)
        self.pos = np.zeros((res, res, 3), np.float32)
        self.nrm = np.zeros((res, res, 3), np.float32)
        self.tan = np.zeros((res, res, 3), np.float32)
        self.bs = np.ones((res, res), np.float32)
        self.mat = np.zeros((res, res), np.int32)
        self.skipped = 0


def rasterize_into(gbuf, tri_uv, attrs, chunk=4096):
    """Rasterize triangles into the g-buffer. `attrs` maps field name ->
    (T, 3, C) or (T, 3) corner data, or ('flat', (T,) data) for per-tri values.
    Generator yielding progress 0..1."""
    res = gbuf.res
    T = len(tri_uv)
    # float64: huge/garbage UVs overflow float32 into inf
    uvpx = tri_uv.astype(np.float64) * res - 0.5
    finite = np.isfinite(uvpx).all(axis=(1, 2))
    n_skipped = int(T - finite.sum())
    gbuf.skipped = n_skipped
    if n_skipped:
        print(f"mesh_baker: skipping {n_skipped} triangle(s) with "
              "non-finite UV coordinates")
    eps = 1e-6

    for start in range(0, T, chunk):
        for i in range(start, min(start + chunk, T)):
            if not finite[i]:
                continue
            a, b, c = uvpx[i]
            fminx = min(a[0], b[0], c[0])
            fmaxx = max(a[0], b[0], c[0])
            fminy = min(a[1], b[1], c[1])
            fmaxy = max(a[1], b[1], c[1])
            # entirely outside the 0..1 tile (or absurd coords): skip
            if fmaxx < 0 or fminx > res - 1 or fmaxy < 0 or fminy > res - 1:
                continue
            minx = int(np.floor(max(fminx, 0.0)))
            miny = int(np.floor(max(fminy, 0.0)))
            maxx = int(np.ceil(min(fmaxx, float(res - 1))))
            maxy = int(np.ceil(min(fmaxy, float(res - 1))))
            if minx > maxx or miny > maxy:
                continue
            d0 = b - a
            d1 = c - a
            det = d0[0] * d1[1] - d0[1] * d1[0]
            if abs(det) < 1e-12:
                continue
            xs = np.arange(minx, maxx + 1, dtype=np.float32)
            ys = np.arange(miny, maxy + 1, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            px = gx - a[0]
            py = gy - a[1]
            w1 = (px * d1[1] - py * d1[0]) / det
            w2 = (-px * d0[1] + py * d0[0]) / det
            w0 = 1.0 - w1 - w2
            inside = (w0 >= -eps) & (w1 >= -eps) & (w2 >= -eps)
            if not inside.any():
                continue
            iy = (gy[inside] + 0.0).astype(np.int64)
            ix = (gx[inside] + 0.0).astype(np.int64)
            W0 = w0[inside][:, None]
            W1 = w1[inside][:, None]
            W2 = w2[inside][:, None]
            gbuf.mask[iy, ix] = True
            for field, data in attrs.items():
                target = getattr(gbuf, field)
                if isinstance(data, tuple) and data[0] == "flat":
                    target[iy, ix] = data[1][i]
                else:
                    corners = data[i]
                    if corners.ndim == 1:  # (3,) scalar per corner
                        val = (W0[:, 0] * corners[0] + W1[:, 0] * corners[1]
                               + W2[:, 0] * corners[2])
                    else:                  # (3, C)
                        val = W0 * corners[0] + W1 * corners[1] + W2 * corners[2]
                    target[iy, ix] = val
        yield min(1.0, (start + chunk) / max(T, 1))
