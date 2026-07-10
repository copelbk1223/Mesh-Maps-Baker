# Mesh Baker

A Blender addon that bakes mesh maps the way Substance Painter / Marmoset Toolbag
do it — with a dedicated texel-space raycast baker, not Blender's render engine.

**Maps:** Tangent Normal (MikkTSpace, OpenGL/DirectX), Ambient Occlusion,
Curvature (derived from the normal map, Substance-style), Thickness,
Position, World Space Normal, Material/Mesh ID.

**Modes:** High poly → Low poly projection (with cage, max frontal/rear
distance, match-by-name) or Single Object.

**Engines:** a compiled C++/Embree core (fast, multithreaded — built
automatically by GitHub Actions) with a pure Python/NumPy fallback that works
anywhere, just slower.

---

## Getting the addon (no compiler needed)

1. Push this repository to GitHub (see below).
2. Open the repo's **Actions** tab — the "Build addon" workflow starts on
   every push.
3. When it finishes, open the run and download the artifact for your OS,
   e.g. `mesh-baker-windows-x64`. Inside is `mesh_baker-windows-x64.zip`.
4. In Blender: `Edit > Preferences > Add-ons > Install…` (in 4.2+: the
   dropdown arrow > *Install from Disk*), pick the zip, enable **Mesh Baker**.
5. The panel appears in the 3D Viewport sidebar (press `N`) under **Mesh Baker**.

The panel shows whether the native core loaded. If it says *Python fallback*,
baking still works but AO/Thickness will be much slower.

## Pushing to GitHub from the editor (VS Code / Antigravity)

1. Install Git for Windows (git-scm.com), then open this folder in the editor.
2. Source Control panel (Ctrl+Shift+G) → **Initialize Repository**.
3. Write a message → **Commit** (stage all when asked).
4. **Publish Branch** → sign in to GitHub → publish as a **public** repository.
5. Every later change: Commit → **Sync Changes**.

Tagging a version (`git tag v1.0 && git push --tags`) attaches the addon zips
to a GitHub Release automatically.

## Usage

**High to Low:** select the high poly mesh(es), then shift-click the low poly
last so it's active. Set mode to *High to Low*, tick the maps, set the output
folder, press **Bake Maps**. Progress shows in the status bar; Esc cancels.

**Single Object:** select one mesh, set mode to *Single Object*, bake.

Tips:
- *Max Frontal / Max Rear* are percentages of the bounding-box diagonal
  (Substance's two-distance model). Raise them if you get misses.
- *Match By Name* pairs `thing_low` with `thing_high` (`_lp/_hp` also work)
  so overlapping parts don't project onto each other.
- A *Cage* object (same topology + UVs as the low poly, inflated) gives
  Marmoset-style controlled projection.
- *Half Res AO/Thickness* traces rays at half resolution and upscales —
  large speedup, on by default.
- Position always saves as 32-bit EXR. Other data maps save as PNG or EXR.
- On low-RAM machines keep high polys under ~2M triangles (decimate the
  sculpt slightly first).

## Building locally (optional)

Requires CMake 3.21+, a C++17 compiler, and Python 3.11 (must match
Blender's bundled Python):

```
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
python scripts/package.py --build-dir build --platform windows-x64 --out dist
```

Embree and pybind11 are downloaded automatically by CMake.

## Project layout

```
addon/mesh_baker/     the Blender addon (Python)
  bake_pipeline.py    bake stages: rasterize -> project -> maps -> save
  mesh_data.py        mesh extraction + UV rasterizer (NumPy)
  engine_python.py    fallback ray engine (mathutils BVHTree)
  engine_native.py    loader for the compiled core
  maps.py             curvature, dilation, sampling, resampling math
cpp/                  the native core (pybind11 + Embree)
scripts/package.py    builds the installable zips
.github/workflows/    CI that compiles for Windows / Linux / macOS
```
