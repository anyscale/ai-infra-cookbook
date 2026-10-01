#!/usr/bin/env python3
"""Runtime patches for sgl_kernel incompatibility on A10G (SM86).

sgl_kernel from PyPI only ships SM100 (Blackwell) CUDA ops.  On A10G (SM86)
the C++ operators never load.  This causes two classes of errors:

1. Import-time: @torch.library.register_fake("sgl_kernel::...") crashes with
   "RuntimeError: operator ... does not exist"
   → Fix: Wrap _register_fake in try/except in torch/library.py

2. Runtime: SGLang's forward_cuda calls torch.ops.sgl_kernel.rmsnorm which
   doesn't exist → "AttributeError: no attribute 'rmsnorm'"
   → Fix: Redirect MultiPlatformOp.dispatch_forward to forward_native

This script applies both patches on all GPU worker nodes via Ray.
"""
import ray


@ray.remote(num_cpus=0.01)
def patch_node():
    """Apply all patches on the current node."""
    import pathlib
    import importlib.util
    import socket

    hostname = socket.gethostname()
    patched = []

    # ---- Locate torch ----
    torch_spec = importlib.util.find_spec("torch")
    if not torch_spec or not torch_spec.submodule_search_locations:
        return f"[{hostname}] torch not found, skipping"
    torch_dir = pathlib.Path(torch_spec.submodule_search_locations[0])

    # ==== Patch 1: torch/library.py ====
    # Wrap _register_fake call with try/except to handle "does not exist" errors
    library_py = torch_dir / "library.py"
    if library_py.exists():
        src = library_py.read_text()
        if "_rf_err" in src:
            patched.append("library.py (already patched)")
        else:
            old = "        use_lib._register_fake(op_name, func, _stacklevel=stacklevel + 1)\n        return func"
            new = (
                "        try:\n"
                "            use_lib._register_fake(op_name, func, _stacklevel=stacklevel + 1)\n"
                "        except RuntimeError as _rf_err:\n"
                '            if "does not exist" in str(_rf_err):\n'
                "                pass  # operator not registered (e.g., sgl_kernel on incompatible arch)\n"
                "            else:\n"
                "                raise\n"
                "        return func"
            )
            if old in src:
                src = src.replace(old, new)
                library_py.write_text(src)
                # Clear bytecode cache
                for pyc in (torch_dir / "__pycache__").glob("library*"):
                    pyc.unlink()
                patched.append("library.py")
            else:
                patched.append("library.py (target not found)")

    # ==== Patch 2: sglang multi_platform.py ====
    # Redirect dispatch_forward to forward_native on CUDA (sgl_kernel ops unavailable)
    sglang_spec = importlib.util.find_spec("sglang")
    if sglang_spec and sglang_spec.submodule_search_locations:
        sglang_dir = pathlib.Path(sglang_spec.submodule_search_locations[0])
        mp_py = sglang_dir / "srt" / "layers" / "utils" / "multi_platform.py"
        if mp_py.exists():
            src = mp_py.read_text()
            if "sgl_kernel ops unavailable" in src:
                patched.append("multi_platform.py (already patched)")
            else:
                old_mp = (
                    "        if _is_cuda:\n"
                    "            return self.forward_cuda"
                )
                new_mp = (
                    "        if _is_cuda:\n"
                    "            return self.forward_native  # sgl_kernel ops unavailable on A10G (SM86)"
                )
                if old_mp in src:
                    src = src.replace(old_mp, new_mp)
                    mp_py.write_text(src)
                    # Clear bytecode cache
                    pycache = mp_py.parent / "__pycache__"
                    if pycache.exists():
                        for pyc in pycache.glob("multi_platform*"):
                            pyc.unlink()
                    patched.append("multi_platform.py")
                else:
                    patched.append("multi_platform.py (target not found)")
        else:
            patched.append("multi_platform.py (file not found)")

    return f"[{hostname}] patched: {', '.join(patched)}"


def main():
    ray.init()

    # Only patch GPU nodes — the head node (CPU-only) can't schedule tasks
    nodes = [
        n for n in ray.nodes()
        if n["Alive"] and n.get("Resources", {}).get("GPU", 0) > 0
    ]
    print(f"Patching {len(nodes)} GPU nodes...")

    refs = []
    for node in nodes:
        ip = node["NodeManagerAddress"]
        ref = patch_node.options(
            resources={f"node:{ip}": 0.001}
        ).remote()
        refs.append(ref)

    results = ray.get(refs)
    for r in results:
        print(f"  {r}")

    print("All nodes patched!")


if __name__ == "__main__":
    main()
