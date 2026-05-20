from __future__ import annotations

import ctypes
import os
from functools import lru_cache


CUDA_REQUIRED_DLLS = [
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cufft64_11.dll",
    "cudart64_12.dll",
    "cudnn64_9.dll",
]


def preload_onnxruntime_dlls() -> None:
    if not cuda_runtime_available():
        return

    try:
        import onnxruntime as ort
    except ImportError:
        return

    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return

    try:
        preload()
    except Exception:
        # Missing CUDA DLLs are reported again when providers are initialized.
        return


@lru_cache(maxsize=1)
def missing_cuda_dlls() -> tuple[str, ...]:
    if os.name != "nt":
        return ()

    missing: list[str] = []
    for dll_name in CUDA_REQUIRED_DLLS:
        try:
            ctypes.WinDLL(dll_name)
        except OSError:
            missing.append(dll_name)
    return tuple(missing)


def cuda_runtime_available() -> bool:
    return not missing_cuda_dlls()


def resolve_providers(requested: list[str]) -> list[str]:
    if "CUDAExecutionProvider" not in requested:
        return requested

    if cuda_runtime_available():
        return requested

    return [provider for provider in requested if provider != "CUDAExecutionProvider"]
