// Mesh Baker native core
// Embree-accelerated ray engine for the Blender Mesh Baker addon.
// Python (the addon) rasterizes the UV g-buffer with NumPy; this module
// does the ray-heavy work: high->low projection, ambient occlusion and
// thickness, multithreaded.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <embree4/rtcore.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

namespace py = pybind11;

// ------------------------------------------------------------ small math
struct V3 { float x, y, z; };
static inline V3 vadd(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V3 vsub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V3 vmul(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline float vdot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 vnorm(V3 a) {
    float l = std::sqrt(vdot(a, a));
    return l > 1e-20f ? vmul(a, 1.0f / l) : V3{0, 0, 1};
}

// Branchless orthonormal basis (Duff et al. 2017)
static inline void onb(V3 n, V3& t, V3& b) {
    float s = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (s + n.z);
    float v = n.x * n.y * a;
    t = {1.0f + s * n.x * n.x * a, s * v, -s * n.x};
    b = {v, s + n.y * n.y * a, -n.y};
}

// PCG-ish RNG, one instance per texel for deterministic bakes
struct RNG {
    uint64_t state;
    explicit RNG(uint64_t seed) {
        state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        next();
    }
    uint32_t next() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t xs = (uint32_t)(((old >> 18u) ^ old) >> 27u);
        uint32_t rot = (uint32_t)(old >> 59u);
        return (xs >> rot) | (xs << ((32u - rot) & 31u));
    }
    float uniform() { return (next() >> 8) * (1.0f / 16777216.0f); }
};

// Cosine-weighted sample inside a cone of half-angle `sinMax` around +Z
static inline V3 sample_cone(RNG& rng, float sinMax) {
    float r1 = rng.uniform(), r2 = rng.uniform();
    float r = std::sqrt(r1) * sinMax;
    float phi = 6.28318530718f * r2;
    float z = std::sqrt(std::max(0.0f, 1.0f - r * r));
    return {r * std::cos(phi), r * std::sin(phi), z};
}

static inline float smoothstep01(float x) {
    x = std::min(std::max(x, 0.0f), 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

// ------------------------------------------------------------ threading
static void parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn) {
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    if (n < 4096 || nt == 1) { fn(0, n); return; }
    std::atomic<size_t> cursor{0};
    const size_t chunk = 2048;
    auto worker = [&]() {
        for (;;) {
            size_t s = cursor.fetch_add(chunk);
            if (s >= n) return;
            fn(s, std::min(n, s + chunk));
        }
    };
    std::vector<std::thread> ts;
    ts.reserve(nt);
    for (unsigned i = 0; i < nt; ++i) ts.emplace_back(worker);
    for (auto& t : ts) t.join();
}

// ------------------------------------------------------------ scene
class Scene {
public:
    Scene(py::array_t<float, py::array::c_style | py::array::forcecast> verts,
          py::array_t<uint32_t, py::array::c_style | py::array::forcecast> tris,
          py::array_t<int32_t, py::array::c_style | py::array::forcecast> tri_mesh) {
        auto v = verts.unchecked<2>();
        auto t = tris.unchecked<2>();
        auto tm = tri_mesh.unchecked<1>();
        const size_t nv = (size_t)v.shape(0);
        const size_t nt = (size_t)t.shape(0);

        verts_.resize(nv);
        for (size_t i = 0; i < nv; ++i) verts_[i] = {v(i, 0), v(i, 1), v(i, 2)};
        tris_.resize(nt * 3);
        for (size_t i = 0; i < nt; ++i) {
            tris_[i * 3 + 0] = t(i, 0);
            tris_[i * 3 + 1] = t(i, 1);
            tris_[i * 3 + 2] = t(i, 2);
        }
        tri_mesh_.resize(nt);
        for (size_t i = 0; i < nt; ++i) tri_mesh_[i] = tm(i);

        device_ = rtcNewDevice(nullptr);
        scene_ = rtcNewScene(device_);
        rtcSetSceneBuildQuality(scene_, RTC_BUILD_QUALITY_HIGH);
        RTCGeometry g = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);
        float* vb = (float*)rtcSetNewGeometryBuffer(
            g, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), nv);
        std::memcpy(vb, verts_.data(), nv * 3 * sizeof(float));
        uint32_t* ib = (uint32_t*)rtcSetNewGeometryBuffer(
            g, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(uint32_t), nt);
        std::memcpy(ib, tris_.data(), nt * 3 * sizeof(uint32_t));
        rtcCommitGeometry(g);
        rtcAttachGeometry(scene_, g);
        rtcReleaseGeometry(g);
        rtcCommitScene(scene_);
    }

    ~Scene() {
        if (scene_) rtcReleaseScene(scene_);
        if (device_) rtcReleaseDevice(device_);
    }
    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    struct Hit {
        int tri = -1;
        float t = 0, u = 0, v = 0;
        V3 ng{0, 0, 1};
    };

    bool cast(V3 o, V3 d, float tnear, float tfar, Hit& out) const {
        RTCRayHit rh;
        rh.ray.org_x = o.x; rh.ray.org_y = o.y; rh.ray.org_z = o.z;
        rh.ray.dir_x = d.x; rh.ray.dir_y = d.y; rh.ray.dir_z = d.z;
        rh.ray.tnear = tnear;
        rh.ray.tfar = tfar;
        rh.ray.time = 0.0f;
        rh.ray.mask = 0xFFFFFFFFu;
        rh.ray.id = 0;
        rh.ray.flags = 0;
        rh.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
        rtcIntersect1(scene_, &rh);
        if (rh.hit.geomID == RTC_INVALID_GEOMETRY_ID) return false;
        out.tri = (int)rh.hit.primID;
        out.t = rh.ray.tfar;
        out.u = rh.hit.u;
        out.v = rh.hit.v;
        out.ng = vnorm({rh.hit.Ng_x, rh.hit.Ng_y, rh.hit.Ng_z});
        return true;
    }

    // First hit along d whose mesh is allowed for the low mesh `lmesh`.
    bool first_allowed(V3 o, V3 d, float tmax, float eps, int lmesh,
                       const uint8_t* allow, int n_high, Hit& out) const {
        float tn = eps;
        for (int i = 0; i < 24; ++i) {
            Hit h;
            if (!cast(o, d, tn, tmax, h)) return false;
            int tm = tri_mesh_[h.tri];
            if (!allow || allow[(size_t)lmesh * n_high + tm]) { out = h; return true; }
            tn = h.t + std::max(eps, h.t * 1e-5f);
            if (tn >= tmax) return false;
        }
        return false;
    }

    // -------------------------------------------------------- projection
    // mode 0: origins = low-poly surface points, dirs = smooth normals.
    //         Search +frontal / -rear, nearest hit to the surface wins.
    // mode 1: cage. origins = cage points, dirs = unit vector cage->surface,
    //         cage_len = |surface - cage|. First hit within len+rear wins.
    py::tuple project(py::array_t<float, py::array::c_style | py::array::forcecast> origins,
                      py::array_t<float, py::array::c_style | py::array::forcecast> dirs,
                      float frontal, float rear, int mode,
                      py::array_t<float, py::array::c_style | py::array::forcecast> cage_len,
                      py::array_t<int32_t, py::array::c_style | py::array::forcecast> tex_mesh,
                      py::array_t<uint8_t, py::array::c_style | py::array::forcecast> allow,
                      float eps) const {
        auto O = origins.unchecked<2>();
        auto D = dirs.unchecked<2>();
        const size_t n = (size_t)O.shape(0);

        const bool has_allow = allow.size() > 0;
        const int n_high = has_allow ? (int)allow.shape(1) : 0;
        const uint8_t* allow_p = has_allow ? allow.data() : nullptr;
        const int32_t* tmesh_p = tex_mesh.size() > 0 ? tex_mesh.data() : nullptr;
        const float* clen_p = cage_len.size() > 0 ? cage_len.data() : nullptr;

        py::array_t<int32_t> r_tri(n);
        py::array_t<float> r_t(n), r_u(n), r_v(n);
        py::array_t<float> r_pos({(py::ssize_t)n, (py::ssize_t)3});
        int32_t* tri_o = r_tri.mutable_data();
        float* t_o = r_t.mutable_data();
        float* u_o = r_u.mutable_data();
        float* v_o = r_v.mutable_data();
        float* p_o = r_pos.mutable_data();

        {
            py::gil_scoped_release release;
            parallel_for(n, [&](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i) {
                    V3 o{O(i, 0), O(i, 1), O(i, 2)};
                    V3 d{D(i, 0), D(i, 1), D(i, 2)};
                    int lm = tmesh_p ? tmesh_p[i] : 0;
                    Hit best;
                    bool found = false;
                    float signed_t = 0.0f;
                    if (mode == 1) {
                        float len = clen_p ? clen_p[i] : 0.0f;
                        Hit h;
                        if (first_allowed(o, d, len + rear, eps, lm, allow_p, n_high, h)) {
                            best = h; found = true; signed_t = len - h.t;
                        }
                    } else {
                        Hit hf, hb;
                        bool ff = frontal > 0.0f &&
                                  first_allowed(o, d, frontal, eps, lm, allow_p, n_high, hf);
                        V3 nd = vmul(d, -1.0f);
                        bool fb = rear > 0.0f &&
                                  first_allowed(o, nd, rear, eps, lm, allow_p, n_high, hb);
                        if (ff && (!fb || hf.t <= hb.t)) {
                            best = hf; found = true; signed_t = hf.t;
                        } else if (fb) {
                            best = hb; found = true; signed_t = -hb.t;
                            // position along -d
                            best.t = hb.t;
                        }
                        if (found && signed_t < 0.0f) d = nd;
                    }
                    if (found) {
                        tri_o[i] = best.tri;
                        t_o[i] = signed_t;
                        u_o[i] = best.u;
                        v_o[i] = best.v;
                        V3 hp = vadd(o, vmul(d, best.t));
                        p_o[i * 3 + 0] = hp.x;
                        p_o[i * 3 + 1] = hp.y;
                        p_o[i * 3 + 2] = hp.z;
                    } else {
                        tri_o[i] = -1;
                        t_o[i] = 0; u_o[i] = 0; v_o[i] = 0;
                        p_o[i * 3 + 0] = o.x; p_o[i * 3 + 1] = o.y; p_o[i * 3 + 2] = o.z;
                    }
                }
            });
        }
        return py::make_tuple(r_tri, r_t, r_u, r_v, r_pos);
    }

    // -------------------------------------------------------- occlusion
    // atten: 0 none, 1 linear, 2 smooth
    // self_mode: 0 always, 1 only same mesh occludes, 2 never self-occlude
    py::array_t<float> occlusion(
        py::array_t<float, py::array::c_style | py::array::forcecast> points,
        py::array_t<float, py::array::c_style | py::array::forcecast> normals,
        int nrays, float maxdist, float spread_deg, int atten,
        bool ignore_backface, int self_mode,
        py::array_t<int32_t, py::array::c_style | py::array::forcecast> pt_mesh,
        uint64_t seed, float eps) const {
        auto P = points.unchecked<2>();
        auto N = normals.unchecked<2>();
        const size_t n = (size_t)P.shape(0);
        const int32_t* pm = pt_mesh.size() > 0 ? pt_mesh.data() : nullptr;

        const float tfar = maxdist > 0.0f ? maxdist : 1e30f;
        const float md = maxdist > 0.0f ? maxdist : 0.0f;
        const float sinMax =
            std::sin(std::min(std::max(spread_deg, 1.0f), 180.0f) * 0.5f * 0.01745329252f);

        py::array_t<float> out(n);
        float* ao = out.mutable_data();

        {
            py::gil_scoped_release release;
            parallel_for(n, [&](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i) {
                    V3 p{P(i, 0), P(i, 1), P(i, 2)};
                    V3 nn = vnorm({N(i, 0), N(i, 1), N(i, 2)});
                    V3 t, b;
                    onb(nn, t, b);
                    RNG rng(seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1)));
                    int pmesh = pm ? pm[i] : 0;
                    V3 o = vadd(p, vmul(nn, eps));
                    float occ = 0.0f;
                    for (int r = 0; r < nrays; ++r) {
                        V3 l = sample_cone(rng, sinMax);
                        V3 d = vnorm(vadd(vadd(vmul(t, l.x), vmul(b, l.y)), vmul(nn, l.z)));
                        float tn = eps;
                        for (int skip = 0; skip < 12; ++skip) {
                            Hit h;
                            if (!cast(o, d, tn, tfar, h)) break;
                            bool backface = vdot(d, h.ng) > 0.0f;
                            int tm = tri_mesh_[h.tri];
                            bool counts = true;
                            if (ignore_backface && backface) counts = false;
                            if (self_mode == 1 && tm != pmesh) counts = false;
                            if (self_mode == 2 && tm == pmesh) counts = false;
                            if (!counts) {
                                tn = h.t + std::max(eps, h.t * 1e-5f);
                                if (tn >= tfar) break;
                                continue;
                            }
                            float w = 1.0f;
                            if (md > 0.0f) {
                                float x = 1.0f - h.t / md;
                                if (atten == 1) w = std::max(0.0f, x);
                                else if (atten == 2) w = smoothstep01(x);
                            }
                            occ += w;
                            break;
                        }
                    }
                    ao[i] = 1.0f - occ / (float)nrays;
                }
            });
        }
        return out;
    }

    // -------------------------------------------------------- thickness
    py::array_t<float> thickness(
        py::array_t<float, py::array::c_style | py::array::forcecast> points,
        py::array_t<float, py::array::c_style | py::array::forcecast> normals,
        int nrays, float maxdist, float spread_deg, uint64_t seed, float eps) const {
        auto P = points.unchecked<2>();
        auto N = normals.unchecked<2>();
        const size_t n = (size_t)P.shape(0);
        const float md = std::max(maxdist, 1e-8f);
        const float sinMax =
            std::sin(std::min(std::max(spread_deg, 1.0f), 180.0f) * 0.5f * 0.01745329252f);

        py::array_t<float> out(n);
        float* th = out.mutable_data();

        {
            py::gil_scoped_release release;
            parallel_for(n, [&](size_t s, size_t e) {
                for (size_t i = s; i < e; ++i) {
                    V3 p{P(i, 0), P(i, 1), P(i, 2)};
                    V3 nn = vnorm({N(i, 0), N(i, 1), N(i, 2)});
                    V3 inward = vmul(nn, -1.0f);
                    V3 t, b;
                    onb(inward, t, b);
                    RNG rng(seed ^ (0xD1B54A32D192ED03ULL * (uint64_t)(i + 1)));
                    V3 o = vadd(p, vmul(inward, eps));
                    float acc = 0.0f;
                    for (int r = 0; r < nrays; ++r) {
                        V3 l = sample_cone(rng, sinMax);
                        V3 d = vnorm(vadd(vadd(vmul(t, l.x), vmul(b, l.y)), vmul(inward, l.z)));
                        Hit h;
                        float dist = md;
                        if (cast(o, d, eps, md, h)) dist = h.t;
                        acc += dist / md;
                    }
                    th[i] = acc / (float)nrays;
                }
            });
        }
        return out;
    }

private:
    RTCDevice device_ = nullptr;
    RTCScene scene_ = nullptr;
    std::vector<V3> verts_;
    std::vector<uint32_t> tris_;
    std::vector<int32_t> tri_mesh_;
};

PYBIND11_MODULE(mesh_baker_core, m) {
    m.doc() = "Mesh Baker native ray engine (Embree)";
    m.attr("__version__") = "1.0.0";
    py::class_<Scene>(m, "Scene")
        .def(py::init<py::array_t<float, py::array::c_style | py::array::forcecast>,
                      py::array_t<uint32_t, py::array::c_style | py::array::forcecast>,
                      py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(),
             py::arg("verts"), py::arg("tris"), py::arg("tri_mesh"))
        .def("project", &Scene::project,
             py::arg("origins"), py::arg("dirs"), py::arg("frontal"), py::arg("rear"),
             py::arg("mode"), py::arg("cage_len"), py::arg("tex_mesh"), py::arg("allow"),
             py::arg("eps"))
        .def("occlusion", &Scene::occlusion,
             py::arg("points"), py::arg("normals"), py::arg("nrays"), py::arg("maxdist"),
             py::arg("spread_deg"), py::arg("atten"), py::arg("ignore_backface"),
             py::arg("self_mode"), py::arg("pt_mesh"), py::arg("seed"), py::arg("eps"))
        .def("thickness", &Scene::thickness,
             py::arg("points"), py::arg("normals"), py::arg("nrays"), py::arg("maxdist"),
             py::arg("spread_deg"), py::arg("seed"), py::arg("eps"));
}
