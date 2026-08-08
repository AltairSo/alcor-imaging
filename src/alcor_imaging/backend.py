"""Optional execution backends.

CuPy is imported lazily so the base package remains lightweight and works on
machines without CUDA. Public processing functions return NumPy arrays; device
arrays are an implementation detail and are released between bounded tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

Backend: TypeAlias = Literal["cpu", "gpu", "auto"]


class GPUUnavailableError(RuntimeError):
    """Raised when GPU execution was requested but CUDA is unavailable."""


@dataclass(frozen=True, slots=True)
class BackendInfo:
    requested: Backend
    selected: Literal["cpu", "gpu"]
    gpu_available: bool
    device_name: str | None = None
    total_memory_bytes: int | None = None


def _import_cupy() -> Any:
    try:
        import cupy as cp
    except (ImportError, OSError) as error:
        raise GPUUnavailableError(
            "GPU processing requires a CUDA-capable NVIDIA GPU and the optional "
            "dependency. Install with `pip install 'alcor-imaging[gpu]'`."
        ) from error
    return cp


def gpu_available() -> bool:
    """Return whether CuPy can access at least one CUDA device."""
    try:
        cp = _import_cupy()
        return bool(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return False


def resolve_backend(backend: Backend = "auto") -> Literal["cpu", "gpu"]:
    """Resolve ``auto`` and validate explicitly requested GPU execution."""
    if backend not in {"cpu", "gpu", "auto"}:
        raise ValueError("backend must be 'cpu', 'gpu', or 'auto'.")
    if backend == "cpu":
        return "cpu"
    available = gpu_available()
    if backend == "gpu" and not available:
        # Re-import to produce the most useful installation/runtime message.
        cp = _import_cupy()
        try:
            count = cp.cuda.runtime.getDeviceCount()
        except Exception as error:
            raise GPUUnavailableError(
                "CuPy is installed, but CUDA could not initialize a GPU. In Colab, "
                "select an NVIDIA GPU runtime and restart the session."
            ) from error
        if count == 0:
            raise GPUUnavailableError("CUDA reported no available NVIDIA GPU devices.")
    return "gpu" if available else "cpu"


def backend_info(backend: Backend = "auto") -> BackendInfo:
    """Describe the selected execution backend and CUDA device."""
    selected = resolve_backend(backend)
    if selected == "cpu":
        return BackendInfo(backend, "cpu", gpu_available())
    cp = _import_cupy()
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties.get("name", properties.get(b"name", b"CUDA GPU"))
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    total = int(cp.cuda.runtime.memGetInfo()[1])
    return BackendInfo(backend, "gpu", True, str(name), total)


def get_array_module(backend: Backend = "auto") -> Any:
    """Return NumPy or CuPy for internal backend-neutral kernels."""
    if resolve_backend(backend) == "cpu":
        import numpy as np

        return np
    return _import_cupy()


def to_host(array: Any) -> Any:
    """Return a NumPy representation of a NumPy or CuPy array."""
    try:
        cp = _import_cupy()
    except GPUUnavailableError:
        return array
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) else array


def clear_gpu_cache() -> None:
    """Release unused CuPy memory-pool blocks back to the CUDA allocator."""
    cp = _import_cupy()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
