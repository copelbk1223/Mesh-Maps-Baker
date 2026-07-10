"""The bake pipeline, mirroring the Substance/Marmoset baker stages:

1. Rasterize the low poly into a texel g-buffer (position/normal/tangent per texel)
2. Project rays from each texel onto the high poly (or use the surface itself)
3. Evaluate each map from the hit data
4. Resolve supersampling, dilate, save

Implemented as a generator yielding (progress, label) so the modal
operator can keep the UI alive and support cancel.
"""
import os
import re
import time

import bpy
import numpy as np

from . import engine_native
from .engine_python import PyScene
from .maps import (
    BakeError, curvature_from_geometry, curvature_from_normal, dilate,
    downsample, gather_hit_attrs,
    id_color, normalize_rows, upsample2,
)
from .mesh_data import GBuffer, merge_high, object_arrays, rasterize_into

PROJ_CHUNK_NATIVE = 262144
PROJ_CHUNK_PY = 1024
RAY_CHUNK_NATIVE = 131072
RAY_CHUNK_PY = 256

_SUFFIX_RE = re.compile(r"\.\d+$")
_LOWHIGH_RE = re.compile(r"[_\.\- ](low|high|lp|hp|lowpoly|highpoly)$", re.IGNORECASE)


def _base_name(name):
    name = _SUFFIX_RE.sub("", name)
    return _LOWHIGH_RE.sub("", name).lower()


def _gather_high_objects(context, s, low):
    if s.high_source == 'COLLECTION':
        if s.high_collection is None:
            raise BakeError("High to Low mode: pick a high poly collection")
        objs = [o for o in s.high_collection.all_objects if o.type == 'MESH']
    else:
        objs = [o for o in context.selected_objects if o.type == 'MESH' and o != low]
    objs = [o for o in objs if o != s.cage_object and o != low]
    if not objs:
        raise BakeError("High to Low mode: no high poly meshes found "
                        "(select them together with the low poly, or pick a collection)")
    return objs


def _make_scene(s, verts, tris, tri_mesh):
    native = None
    if s.engine in ('AUTO', 'NATIVE'):
        native = engine_native.load()
        if s.engine == 'NATIVE' and native is None:
            raise BakeError("Native engine requested but the compiled core is not "
                            "available for this platform")
    if native is not None:
        return native.Scene(
            np.ascontiguousarray(verts, np.float32),
            np.ascontiguousarray(tris, np.uint32),
            np.ascontiguousarray(tri_mesh, np.int32),
        ), True
    return PyScene(verts, tris, tri_mesh), False


def _save_map(results, s, suffix, rgb, res, colorspace, force_exr=False):
    base = s.base_name.strip() or results["base"]
    name = f"{base}_{suffix}"
    old = bpy.data.images.get(name)
    if old is not None:
        bpy.data.images.remove(old)
    img = bpy.data.images.new(name, res, res, alpha=False, float_buffer=True)
    img.colorspace_settings.name = colorspace
    px = np.ones((res, res, 4), np.float32)
    px[..., :3] = np.clip(rgb, 0.0, None)
    img.pixels.foreach_set(px.ravel())
    if s.save_to_disk:
        fmt = 'OPEN_EXR' if (force_exr or s.fmt_data == 'OPEN_EXR') else 'PNG'
        out_dir = bpy.path.abspath(s.out_dir) or bpy.app.tempdir
        os.makedirs(out_dir, exist_ok=True)
        ext = ".exr" if fmt == 'OPEN_EXR' else ".png"
        img.filepath_raw = os.path.join(out_dir, name + ext)
        img.file_format = fmt
        img.save()
        results["files"].append(img.filepath_raw)
    results["images"].append(name)


def _scatter(shape_res, valid_yx, values, fill=0.0):
    """Scatter flat per-texel values back into an (R, R, C) image."""
    if values.ndim == 1:
        values = values[:, None]
    img = np.full((shape_res, shape_res, values.shape[1]), fill, np.float32)
    img[valid_yx[0], valid_yx[1]] = values
    return img


def _resolve(img, mask, ss, dilation):
    """Supersampling resolve + dilation at final resolution."""
    out, m = downsample(img, mask, ss)
    out, m = dilate(out, m, dilation)
    return out


def run_bake(context, results):
    t0 = time.time()
    s = context.scene.mesh_baker
    low = context.active_object
    if low is None or low.type != 'MESH':
        raise BakeError("The active object must be the low poly mesh")
    results.update({"files": [], "images": [], "warnings": [], "base": low.name})

    deps = context.evaluated_depsgraph_get()
    res = int(s.resolution)
    ss = int(s.aa)
    R = res * ss
    h2l = s.mode == 'H2L'

    # ---------------------------------------------------------- 1. low poly
    yield 0.01, "Reading low poly"
    L = object_arrays(low, deps, need_tangents=True)

    gbuf = GBuffer(R)
    attrs = {
        "pos": L["verts"][L["tri_v"]],
        "nrm": L["tri_nrm"],
        "tan": L["tri_tan"],
        "bs": L["tri_bs"],
        "mat": ("flat", L["tri_mat"]),
    }
    for frac in rasterize_into(gbuf, L["tri_uv"], attrs):
        yield 0.01 + 0.10 * frac, "Rasterizing UVs"
    if gbuf.skipped:
        results["warnings"].append(
            f"{gbuf.skipped} triangle(s) have corrupt UVs (inf/NaN) and were "
            "skipped — re-unwrap the affected faces to fix")
    if not gbuf.mask.any():
        raise BakeError(
            "No texels covered — the UV map is empty, corrupt, or entirely "
            "outside the 0..1 tile. Re-unwrap the low poly and try again")

    yx = np.nonzero(gbuf.mask)
    n = len(yx[0])
    pos = gbuf.pos[yx]
    nrm = normalize_rows(gbuf.nrm[yx]).astype(np.float32)
    tan = normalize_rows(gbuf.tan[yx]).astype(np.float32)
    bs = gbuf.bs[yx]
    low_mat = gbuf.mat[yx]

    # ------------------------------------------------------------- 2. scene
    bb_min = L["verts"].min(0)
    bb_max = L["verts"].max(0)

    if h2l:
        highs = _gather_high_objects(context, s, low)
        datas = []
        for i, ob in enumerate(highs):
            yield 0.11 + 0.06 * (i / len(highs)), f"Reading high poly ({ob.name})"
            off = sum(len(d["mat_names"]) for d in datas)
            datas.append(object_arrays(ob, deps, need_tangents=False,
                                       mat_name_offset=off, mesh_id=i))
        H = merge_high(datas)
        bb_min = np.minimum(bb_min, H["verts"].min(0))
        bb_max = np.maximum(bb_max, H["verts"].max(0))
        yield 0.18, "Building BVH"
        scene, is_native = _make_scene(s, H["verts"], H["tri_v"], H["tri_mesh"])
        id_names = H["mat_names"] if s.id_source == 'MATERIAL' else H["mesh_names"]
    else:
        yield 0.18, "Building BVH"
        scene, is_native = _make_scene(
            s, L["verts"], L["tri_v"].astype(np.uint32),
            np.zeros(len(L["tri_v"]), np.int32))
        id_names = L["mat_names"] if s.id_source == 'MATERIAL' else [L["name"]]

    diag = float(np.linalg.norm(bb_max - bb_min))
    if diag <= 0:
        raise BakeError("Degenerate bounding box")
    eps = 1e-4 * diag
    frontal = s.frontal_pct / 100.0 * diag
    rear = s.rear_pct / 100.0 * diag
    results["engine"] = "native" if is_native else "python"

    # -------------------------------------------------------- 3. projection
    if h2l:
        allow = np.zeros((0, 0), np.uint8)
        tex_mesh = np.zeros(0, np.int32)
        if s.match_by_name:
            lb = _base_name(low.name)
            row = np.array([1 if _base_name(nm) == lb else 0
                            for nm in H["mesh_names"]], np.uint8)
            if row.any():
                allow = row[None, :]
                tex_mesh = np.zeros(n, np.int32)
            else:
                results["warnings"].append(
                    "Match By Name: no high poly matched "
                    f"'{lb}' — matching disabled for this bake")

        mode = 0
        cage_len = np.zeros(0, np.float32)
        origins, dirs = pos, nrm
        if s.cage_object is not None:
            cage_g = GBuffer(R)
            C = object_arrays(s.cage_object, deps, need_tangents=True)
            for frac in rasterize_into(cage_g, C["tri_uv"],
                                       {"pos": C["verts"][C["tri_v"]]}):
                yield 0.19 + 0.03 * frac, "Rasterizing cage"
            cpos = cage_g.pos[yx]
            cvalid = cage_g.mask[yx]
            delta = pos - cpos
            clen = np.linalg.norm(delta, axis=-1)
            good = cvalid & (clen > 1e-9)
            d = np.where(good[:, None], delta / np.maximum(clen[:, None], 1e-12),
                         -nrm)  # fallback: shoot inward from an offset point
            o = np.where(good[:, None], cpos, pos + nrm * frontal)
            cage_len = np.where(good, clen, frontal).astype(np.float32)
            origins, dirs, mode = o.astype(np.float32), d.astype(np.float32), 1
            if not good.all():
                results["warnings"].append(
                    "Cage did not cover all texels — normal-offset fallback used there")

        chunk = PROJ_CHUNK_NATIVE if is_native else PROJ_CHUNK_PY
        hit_tri = np.empty(n, np.int32)
        hit_t = np.empty(n, np.float32)
        hit_u = np.empty(n, np.float32)
        hit_v = np.empty(n, np.float32)
        hit_pos = np.empty((n, 3), np.float32)
        for st in range(0, n, chunk):
            en = min(st + chunk, n)
            tm = tex_mesh[st:en] if len(tex_mesh) else tex_mesh
            cl = cage_len[st:en] if len(cage_len) else cage_len
            r = scene.project(origins[st:en], dirs[st:en], frontal, rear,
                              mode, cl, tm, allow, eps)
            (hit_tri[st:en], hit_t[st:en], hit_u[st:en],
             hit_v[st:en], hit_pos[st:en]) = r
            yield 0.22 + 0.18 * (en / n), f"Projecting rays  {en}/{n}"

        found = hit_tri >= 0
        if s.proj_ignore_backface:
            st_ = np.clip(hit_tri, 0, None)
            tv = H["verts"][H["tri_v"][st_]]
            ng = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
            ray_dir = dirs if mode == 1 else dirs * np.where(
                hit_t >= 0, 1.0, -1.0)[:, None]
            backface = (ray_dir * ng).sum(-1) > 0
            found = found & ~backface
        miss = int(n - found.sum())
        if miss:
            results["warnings"].append(
                f"{miss} texels found no high poly within the ray distance "
                "(increase Max Frontal/Rear)")
        hit_nrm = normalize_rows(
            gather_hit_attrs(hit_tri, hit_u, hit_v, H["tri_nrm"])).astype(np.float32)
        safe_tri = np.clip(hit_tri, 0, None)
        hit_mat = H["tri_mat"][safe_tri]
        hit_mesh = H["tri_mesh"][safe_tri]
        valid = found
    else:
        hit_pos, hit_nrm = pos, nrm
        hit_mat = low_mat
        hit_mesh = np.zeros(n, np.int32)
        hit_t = np.zeros(n, np.float32)
        valid = np.ones(n, bool)
        yield 0.35, "Using surface data"

    vmask2d = np.zeros((R, R), bool)
    vmask2d[yx[0][valid], yx[1][valid]] = True

    # ------------------------------------------------------------- 4. maps
    need_nts = s.use_normal or (
        s.use_curvature and (s.curv_source == 'NORMALMAP'
                             or (s.curv_source == 'AUTO' and h2l)))
    nts_img = None
    if need_nts:
        yield 0.42, "Computing normal map"
        bit = np.cross(nrm, tan) * bs[:, None]
        nts = np.stack([
            (hit_nrm * tan).sum(-1),
            (hit_nrm * bit).sum(-1),
            (hit_nrm * nrm).sum(-1),
        ], -1)
        nts = normalize_rows(nts).astype(np.float32)
        nts_img = _scatter(R, yx, np.where(valid[:, None], nts, [0, 0, 1]))
        nts_img[~gbuf.mask] = (0, 0, 1)
        # dilate raw normal data so curvature derivatives don't pick up seams
        nts_img, _ = dilate(nts_img, vmask2d.copy(), max(4, min(s.dilation, 16)))

    if s.use_normal:
        enc = nts_img.copy()
        if s.normal_directx:
            enc[..., 1] = -enc[..., 1]
        enc = enc * 0.5 + 0.5
        out = _resolve(enc, vmask2d, ss, s.dilation)
        _save_map(results, s, "Normal", out, res, 'Non-Color')
        yield 0.46, "Saved normal map"

    if s.use_curvature:
        yield 0.48, "Computing curvature"
        cmode = s.curv_source
        if cmode == 'AUTO':
            cmode = 'NORMALMAP' if h2l else 'MESH'
        if cmode == 'NORMALMAP' and nts_img is not None:
            curv = curvature_from_normal(nts_img, vmask2d, s.curv_intensity,
                                         s.curv_smooth, s.curv_invert, R)
        else:
            nws_img = _scatter(R, yx, np.where(valid[:, None], hit_nrm, 0))
            pos_img = _scatter(R, yx, hit_pos)
            nws_img, _ = dilate(nws_img, vmask2d.copy(), 4)
            pos_img, _ = dilate(pos_img, vmask2d.copy(), 4)
            curv = curvature_from_geometry(nws_img, pos_img, vmask2d,
                                           s.curv_intensity, s.curv_smooth,
                                           s.curv_invert, diag,
                                           s.curv_auto_tonemap)
        out = _resolve(curv, vmask2d, ss, s.dilation)
        _save_map(results, s, "Curvature", out, res, 'Non-Color')

    if s.use_ws_normal:
        img = _scatter(R, yx, np.where(valid[:, None], hit_nrm * 0.5 + 0.5, 0.5))
        out = _resolve(img, vmask2d, ss, s.dilation)
        _save_map(results, s, "WorldSpaceNormal", out, res, 'Non-Color')
        yield 0.50, "Saved world space normal"

    if s.use_position:
        if s.pos_normalization == 'BSPHERE':
            center = (bb_min + bb_max) * 0.5
            radius = max(float(np.linalg.norm(bb_max - center)), 1e-12)
            p01 = (hit_pos - center) / (2.0 * radius) + 0.5
        else:
            p01 = (hit_pos - bb_min) / np.maximum(bb_max - bb_min, 1e-12)
        if s.pos_mode != 'ALL':
            ax = {'X': 0, 'Y': 1, 'Z': 2}[s.pos_mode]
            p01 = p01[:, ax:ax + 1].repeat(3, axis=1)
        img = _scatter(R, yx, np.clip(p01, 0, 1))
        out = _resolve(img, vmask2d, ss, s.dilation)
        _save_map(results, s, "Position", out, res, 'Non-Color')
        yield 0.52, "Saved position map"

    if s.use_height:
        if not h2l:
            results["warnings"].append(
                "Height map needs High to Low mode (it measures the distance "
                "between the two surfaces) — skipped")
        else:
            if s.height_scale_pct > 0:
                ref = s.height_scale_pct / 100.0 * diag
            else:
                hv = np.abs(hit_t[valid])
                ref = float(np.percentile(hv, 99.0)) if len(hv) else 1.0
            ref = max(ref, 1e-12)
            hh = np.clip(0.5 + hit_t / (2.0 * ref), 0.0, 1.0)
            img = _scatter(R, yx, np.where(valid, hh, 0.5)[:, None].repeat(3, 1))
            out = _resolve(img, vmask2d, ss, s.dilation)
            _save_map(results, s, "Height", out, res, 'Non-Color')
            yield 0.53, "Saved height map"

    if s.use_matid:
        table = np.stack([id_color(nm) for nm in id_names]) if id_names else \
            np.ones((1, 3), np.float32)
        ids = hit_mesh if (h2l and s.id_source == 'MESH') else hit_mat
        cols = table[np.clip(ids, 0, len(table) - 1)]
        img = _scatter(R, yx, np.where(valid[:, None], cols, 0))
        out = _resolve(img, vmask2d, ss, s.dilation)
        _save_map(results, s, "ID", out, res, 'Non-Color')
        yield 0.54, "Saved ID map"

    # ------------------------------------------------- ray maps (AO, thickness)
    def ray_points():
        """Points/normals/mesh ids to trace from, optionally at half res."""
        if s.half_res_rays and R >= 512:
            sel = (yx[0] % 2 == 0) & (yx[1] % 2 == 0)
            hy, hx = yx[0][sel] // 2, yx[1][sel] // 2
            return sel, (hy, hx), R // 2, True
        return np.ones(n, bool), yx, R, False

    ray_chunk = RAY_CHUNK_NATIVE if is_native else RAY_CHUNK_PY

    def trace_map(fn_name, lo, hi, label, **kw):
        sel, syx, sres, half = ray_points()
        pts = hit_pos[sel]
        nms = hit_nrm[sel]
        pmesh = hit_mesh[sel] if h2l else np.zeros(0, np.int32)
        m = len(pts)
        vals = np.ones(m, np.float32)
        for st in range(0, m, ray_chunk):
            en = min(st + ray_chunk, m)
            pm = pmesh[st:en] if len(pmesh) else pmesh
            if fn_name == "occlusion":
                vals[st:en] = scene.occlusion(
                    pts[st:en], nms[st:en], kw["rays"], kw["maxdist"],
                    kw["spread"], kw["atten"], kw["backface"], kw["self_mode"],
                    pm, s.seed + 1, eps)
            else:
                vals[st:en] = scene.thickness(
                    pts[st:en], nms[st:en], kw["rays"], kw["maxdist"],
                    kw["spread"], s.seed + 2, eps)
            yield lo + (hi - lo) * (en / max(m, 1)), f"{label}  {en}/{m}"
        vsel = valid[sel]
        img = _scatter(sres, syx, np.where(vsel, vals, 1.0))
        msk = np.zeros((sres, sres), bool)
        msk[syx[0][vsel], syx[1][vsel]] = True
        if half:
            img, msk = upsample2(img, msk)
            msk = msk & vmask2d
        out = _resolve(img.repeat(3, axis=-1), msk | vmask2d, ss, s.dilation)
        yield ("RESULT", out)

    if s.use_ao:
        if not is_native and n * s.ao_rays > 5e7:
            results["warnings"].append(
                "AO on the Python engine at these settings is very slow — "
                "install the native build or lower resolution/rays")
        atten = {'NONE': 0, 'LINEAR': 1, 'SMOOTH': 2}[s.ao_attenuation]
        self_mode = {'ALWAYS': 0, 'SAME': 1, 'NEVER': 2}[s.ao_self]
        maxd = s.ao_distance_pct / 100.0 * diag
        if atten and maxd <= 0:
            maxd = diag
        out = None
        for item in trace_map("occlusion", 0.55, 0.78, "Ambient occlusion",
                              rays=s.ao_rays, maxdist=maxd, spread=s.ao_spread,
                              atten=atten, backface=s.ao_ignore_backface,
                              self_mode=self_mode):
            if isinstance(item, tuple) and item[0] == "RESULT":
                out = item[1]
            else:
                yield item
        _save_map(results, s, "AO", out, res, 'Non-Color')

    if s.use_thickness:
        maxd = max(s.th_distance_pct, 0.1) / 100.0 * diag
        out = None
        for item in trace_map("thickness", 0.78, 0.97, "Thickness",
                              rays=s.th_rays, maxdist=maxd, spread=s.th_spread):
            if isinstance(item, tuple) and item[0] == "RESULT":
                out = item[1]
            else:
                yield item
        _save_map(results, s, "Thickness", out, res, 'Non-Color')

    results["time"] = time.time() - t0
    yield 1.0, "Done"
