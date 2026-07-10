"""Loader for the compiled C++/Embree core (mesh_baker_core).
Falls back gracefully if no binary is bundled for this platform."""
import os
import platform
import sys

_module = None
_tried = False
_error = ""


def platform_tag():
    if sys.platform == "win32":
        return "windows-x64"
    if sys.platform == "darwin":
        return "macos-arm64" if platform.machine() in ("arm64", "aarch64") else "macos-x64"
    return "linux-x64"


def load():
    """Return the native module or None. Caches the result."""
    global _module, _tried, _error
    if _tried:
        return _module
    _tried = True

    native_dir = os.path.join(os.path.dirname(__file__), "native", platform_tag())
    if not os.path.isdir(native_dir):
        _error = f"no native build for {platform_tag()}"
        return None

    if sys.platform == "win32":
        try:
            os.add_dll_directory(native_dir)
        except (AttributeError, OSError):
            pass
    if native_dir not in sys.path:
        sys.path.insert(0, native_dir)

    try:
        import mesh_baker_core  # noqa: F401
        _module = mesh_baker_core
        print(f"mesh_baker: native core {mesh_baker_core.__version__} loaded")
    except Exception as e:  # keep the addon usable without the binary
        _error = str(e)
        print(f"mesh_baker: native core failed to load ({e}), using Python fallback")
    return _module


def status():
    """(loaded: bool, message: str) for the UI."""
    m = load()
    if m is not None:
        return True, f"Native core v{m.__version__}"
    return False, f"Python fallback ({_error})"
