"""NumPy helpers shared by both engines and the pipeline:
image-space operations (dilation, curvature, resampling) and sampling math.
All image arrays are (H, W, C) float32 with row 0 = UV v=0 (Blender order).
"""
import colorsys
import zlib

import numpy as np


class BakeError(Exception):
    pass


# ------------------------------------------------------------------ vectors
def normalize_rows(a):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(n, 1e-12)


def onb_from_normals(n):
    """Branchless ONB (Duff et al.) for an (N,3) normal array -> (t, b)."""
    s = np.where(n[:, 2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (s + n[:, 2])
    v = n[:, 0] * n[:, 1] * a
    t = np.stack([1.0 + s * n[:, 0] ** 2 * a, s * v, -s * n[:, 0]], axis=-1)
    b = np.stack([v, s + n[:, 1] ** 2 * a, -n[:, 1]], axis=-1)
    return t.astype(np.float32), b.astype(np.float32)


def cosine_cone_samples(nrays, spread_deg, rng):
    """Cosine-weighted directions in a cone around +Z, (nrays, 3)."""
    sin_max = np.sin(np.radians(np.clip(spread_deg, 1.0, 180.0)) * 0.5)
    r1 = rng.random(nrays)
    r2 = rng.random(nrays)
    r = np.sqrt(r1) * sin_max
    phi = 2.0 * np.pi * r2
    z = np.sqrt(np.maximum(0.0, 1.0 - r * r))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], -1).astype(np.float32)


# ------------------------------------------------------------------ images
_NB_SHIFTS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _shift2d(a, dy, dx, fill=0.0):
    out = np.full_like(a, fill)
    h, w = a.shape[:2]
    ys = slice(max(dy, 0), h + min(dy, 0))
    xs = slice(max(dx, 0), w + min(dx, 0))
    yd = slice(max(-dy, 0), h + min(-dy, 0))
    xd = slice(max(-dx, 0), w + min(-dx, 0))
    out[yd, xd] = a[ys, xs]
    return out


def dilate(img, mask, iterations):
    """Flood-fill edge padding outside `mask`, Substance-style dilation."""
    img = img.copy()
    mask = mask.copy()
    for _ in range(iterations):
        if mask.all():
            break
        acc = np.zeros_like(img)
        cnt = np.zeros(mask.shape, np.float32)
        mf = mask.astype(np.float32)
        for dy, dx in _NB_SHIFTS:
            acc += _shift2d(img * mf[..., None], dy, dx)
            cnt += _shift2d(mf, dy, dx)
        new = (cnt > 0) & ~mask
        if not new.any():
            break
        img[new] = acc[new] / cnt[new, None]
        mask = mask | new
    return img, mask


def blur_masked(img, mask, iterations=1):
    out = img.copy()
    mf = mask.astype(np.float32)
    for _ in range(iterations):
        acc = out * mf[..., None]
        cnt = mf.copy()
        for dy, dx in _NB_SHIFTS:
            acc = acc + _shift2d(out * mf[..., None], dy, dx)
            cnt = cnt + _shift2d(mf, dy, dx)
        valid = cnt > 0
        out[valid] = acc[valid] / cnt[valid, None]
    return out


def downsample(img, mask, ss):
    """Mask-weighted box downsample by integer factor ss (supersampling resolve)."""
    if ss == 1:
        return img, mask
    h = img.shape[0] // ss
    w = img.shape[1] // ss
    c = img.shape[2]
    iv = img.reshape(h, ss, w, ss, c)
    mv = mask.reshape(h, ss, w, ss).astype(np.float32)
    s = (iv * mv[:, :, :, :, None]).sum(axis=(1, 3))
    n = mv.sum(axis=(1, 3))
    out = np.zeros((h, w, c), np.float32)
    nz = n > 0
    out[nz] = s[nz] / n[nz, None]
    return out, nz


def upsample2(img, mask):
    """Nearest x2 upsample followed by one masked blur pass (half-res ray maps)."""
    big = np.repeat(np.repeat(img, 2, axis=0), 2, axis=1)
    bm = np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)
    return blur_masked(big, bm, 1), bm


def curvature_from_normal(nrm_ts, mask, intensity, smooth_iters, invert, res):
    """Curvature the Substance/Marmoset way: divergence of the tangent-space
    normal map. Convex = bright, concave = dark, 0.5 = flat.
    nrm_ts: (H, W, 3) in [-1, 1]."""
    n = nrm_ts
    if smooth_iters > 0:
        n = blur_masked(n.copy(), mask, smooth_iters)
    # d(nx)/du + d(ny)/dv ; axis 1 = u (x), axis 0 = v (y)
    gx = np.gradient(n[..., 0], axis=1)
    gy = np.gradient(n[..., 1], axis=0)
    div = (gx + gy) * (res / 1024.0) * 8.0 * intensity
    if invert:
        div = -div
    out = np.clip(0.5 + div * 0.5, 0.0, 1.0).astype(np.float32)
    return out[..., None].repeat(3, axis=-1)


# ------------------------------------------------------------------ id map
def id_color(name):
    """Deterministic bright color from a name (stable across sessions)."""
    h = zlib.crc32(name.encode("utf-8"))
    hue = (h % 3600) / 3600.0
    sat = 0.65 + ((h >> 12) % 35) / 100.0
    val = 0.85 + ((h >> 20) % 15) / 100.0
    return np.array(colorsys.hsv_to_rgb(hue, sat, val), np.float32)


def gather_hit_attrs(hit_tri, hit_u, hit_v, tri_corner):
    """Interpolate per-corner data (T,3,C) at hits using Embree barycentrics
    (u -> corner 1, v -> corner 2)."""
    t = np.clip(hit_tri, 0, None)
    w0 = (1.0 - hit_u - hit_v)[:, None]
    c = tri_corner[t]
    return w0 * c[:, 0] + hit_u[:, None] * c[:, 1] + hit_v[:, None] * c[:, 2]


def barycentric(p, a, b, c):
    """(u, v) barycentrics of points p in triangles abc, Embree convention."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    u = (d11 * d20 - d01 * d21) / den
    v = (d00 * d21 - d01 * d20) / den
    return u.astype(np.float32), v.astype(np.float32)
