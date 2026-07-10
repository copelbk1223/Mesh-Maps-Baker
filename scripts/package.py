"""Package the Mesh Baker addon into an installable zip.

Copies addon/mesh_baker into a staging dir, drops the compiled
mesh_baker_core module + Embree runtime libraries into
mesh_baker/native/<platform>/ and zips the result.

Usage (from repo root):
    python scripts/package.py --build-dir build --platform windows-x64 --out dist
    python scripts/package.py --pure --out dist        # python-only fallback zip
"""
import argparse
import glob
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDON_SRC = os.path.join(ROOT, "addon", "mesh_baker")


def find_module(build_dir):
    pats = ["**/mesh_baker_core*.pyd", "**/mesh_baker_core*.so", "**/mesh_baker_core*.dylib"]
    hits = []
    for p in pats:
        hits += glob.glob(os.path.join(build_dir, p), recursive=True)
    if not hits:
        sys.exit(f"ERROR: compiled mesh_baker_core module not found under {build_dir}")
    return hits[0]


def find_runtime_libs(build_dir, plat):
    """Embree + TBB shared libraries that must ship next to the module."""
    deps = glob.glob(os.path.join(build_dir, "_deps", "embree_bin-src"))
    if not deps:
        sys.exit("ERROR: embree_bin-src not found (FetchContent dir missing)")
    root = deps[0]
    libs = []
    if plat.startswith("windows"):
        libs = glob.glob(os.path.join(root, "bin", "*.dll"))
    elif plat.startswith("linux"):
        libs = glob.glob(os.path.join(root, "lib", "libembree4.so*"))
        libs += glob.glob(os.path.join(root, "lib", "libtbb*.so*"))
    elif plat.startswith("macos"):
        libs = glob.glob(os.path.join(root, "lib", "*.dylib"))
    return [p for p in libs if os.path.isfile(p) and not os.path.islink(p)] + \
           [p for p in libs if os.path.islink(p)]


def stage_addon(staging):
    dst = os.path.join(staging, "mesh_baker")
    shutil.copytree(
        ADDON_SRC, dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "native"),
    )
    os.makedirs(os.path.join(dst, "native"), exist_ok=True)
    return dst


def zip_dir(staging, zip_path):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _dirs, files in os.walk(staging):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, os.path.relpath(full, staging))
    print("wrote", zip_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--platform", default="")
    ap.add_argument("--out", default="dist")
    ap.add_argument("--pure", action="store_true", help="package without a native core")
    args = ap.parse_args()

    staging = os.path.join(ROOT, "_staging")
    if os.path.isdir(staging):
        shutil.rmtree(staging)
    dst = stage_addon(staging)

    if args.pure:
        zip_dir(staging, os.path.join(ROOT, args.out, "mesh_baker-python-only.zip"))
        return

    plat = args.platform
    native_dir = os.path.join(dst, "native", plat)
    os.makedirs(native_dir, exist_ok=True)

    module = find_module(os.path.join(ROOT, args.build_dir))
    shutil.copy2(module, native_dir)
    print("core module:", os.path.basename(module))

    for lib in find_runtime_libs(os.path.join(ROOT, args.build_dir), plat):
        target = os.path.join(native_dir, os.path.basename(lib))
        if os.path.islink(lib):
            real = os.path.realpath(lib)
            shutil.copy2(real, target)
        else:
            shutil.copy2(lib, target)
        print("runtime lib:", os.path.basename(lib))

    zip_dir(staging, os.path.join(ROOT, args.out, f"mesh_baker-{plat}.zip"))


if __name__ == "__main__":
    main()
