"""Make only file paths relative; preserve AdaPT physics, policy and task config."""
import importlib
import os
from pathlib import Path


def short_windows_path(path):
    if os.name != "nt":
        return str(path)
    import ctypes
    buffer = ctypes.create_unicode_buffer(32768)
    result = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    if not result:
        raise OSError("Could not obtain the existing Windows short path")
    path = buffer.value.removeprefix("\\\\?\\")
    if not path.isascii():
        raise OSError("NTFS short names unavailable; use an ASCII-only checkout for NVRTC")
    return path


def patch_adapt_paths():
    import warp as wp
    # The pinned Warp 1.16 moved the old public context module under _src.
    if not hasattr(wp, "context"):
        wp.context = importlib.import_module("warp._src.context")
        print("[COMPAT] Restore the Warp context alias used by upstream mjlab.")
    if os.name == "nt":
        # NVRTC cannot resolve the Japanese absolute include path even though
        # Python can. Use existing NTFS short names; do not copy the library.
        build = importlib.import_module("warp._src.build")
        build.warp_home = short_windows_path(build.warp_home)
        from project_paths import CACHE
        cache = CACHE / "warp-cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["WARP_CACHE_PATH"] = short_windows_path(cache)
        # NVRTC's precompiled-header temporary directory must also be ASCII and
        # writable. Windows app sandboxes may provide an inaccessible TEMP path.
        import tempfile
        temp = cache / "tmp"
        temp.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = short_windows_path(temp)
        os.environ["TMP"] = os.environ["TEMP"] = tempfile.tempdir
        wp.init()
        # Warp init resolves short names back to Unicode; set the existing
        # version-specific cache AFTER init so its C file API sees ASCII.
        wp.config.kernel_cache_dir = short_windows_path(wp.config.kernel_cache_dir)
        print("[COMPAT] Use Windows short paths for NVRTC includes and cache.")
    scene = importlib.import_module("mjlab.scene.scene")
    g1 = importlib.import_module("mjlab.asset_zoo.robots.unitree_g1.g1_constants")
    scene._SCENE_XML = Path(os.path.relpath(scene._SCENE_XML))
    g1.G1_LEFT_XML = Path(os.path.relpath(g1.G1_LEFT_XML))
    g1.G1_RIGHT_XML = Path(os.path.relpath(g1.G1_RIGHT_XML))
    print("[COMPAT] Use ASCII relative MuJoCo filenames; upstream remains unchanged.")


if __name__ == "__main__":
    patch_adapt_paths()
    from mjlab.scripts.play import main
    main()
