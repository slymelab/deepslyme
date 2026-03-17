import gc
import random
from typing import Optional
import numpy as np
import torch

_MANUAL_BACKEND: Optional[str] = None


def set_backend(backend_name: Optional[str]) -> None:
    """Manually set the backend"""
    global _MANUAL_BACKEND
    _MANUAL_BACKEND = backend_name


def _get_active_backend_name() -> str:
    """Get the active backend name."""
    if _MANUAL_BACKEND is not None:
        return _MANUAL_BACKEND

    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return str(torch.accelerator.current_accelerator().type)

    # Simple fallback
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    elif hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():
        return "xpu"
    elif hasattr(torch, "npu") and getattr(torch.npu, "is_available", lambda: False)():
        return "npu"

    return "cpu"


def current_device_name() -> str:
    """Get the device backend name."""
    return _get_active_backend_name()


def current_comm_backend_name() -> str:
    """
    Retrieve the recommended distributed communication backend name
    (e.g., 'nccl', 'gloo') based on the active hardware accelerator.

    Returns:
        str: The name of the distributed backend.
             Typically 'nccl' for CUDA and 'gloo' for CPU/MPS.
    """
    backend = _get_active_backend_name()
    # NCCL is the gold standard for NVIDIA GPUs.
    if backend == "cuda":
        return "nccl"
    if backend == "xpu":
        return "ccl"
    if backend == "npu":
        return "hccl"
    return "gloo"


def is_available() -> bool:
    """Check if any accelerator is available."""
    backend = _get_active_backend_name()
    if backend == "cpu":
        return False
    if backend == "cuda":
        return torch.cuda.is_available()
    if backend == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if backend == "xpu":
        return (
            hasattr(torch, "xpu")
            and getattr(torch.xpu, "is_available", lambda: False)()
        )
    if backend == "npu":
        return (
            hasattr(torch, "npu")
            and getattr(torch.npu, "is_available", lambda: False)()
        )
    return False


def current_accelerator() -> torch.device:
    """Get the current accelerator device."""
    backend = _get_active_backend_name()
    return torch.device(backend)


def device_count() -> int:
    """Get the number of available devices."""
    backend = _get_active_backend_name()
    if backend == "cuda":
        return torch.cuda.device_count()
    elif backend == "xpu":
        return torch.xpu.device_count()
    elif backend == "npu":
        return torch.npu.device_count()
    elif backend == "mps":
        return 1
    return 0


def set_device_index(device_index: int) -> None:
    """Set the device index for the current accelerator."""
    backend = _get_active_backend_name()
    if backend == "cuda":
        torch.cuda.set_device(device_index)
    elif backend == "xpu":
        torch.xpu.set_device(device_index)
    elif backend == "npu":
        torch.npu.set_device(device_index)


def empty_cache() -> None:
    """Empty the cache of the current accelerator."""
    gc.collect()
    backend = _get_active_backend_name()

    if hasattr(torch, "accelerator") and hasattr(torch.accelerator, "empty_cache"):
        # Call the official empty_cache if available
        torch.accelerator.empty_cache()
    elif backend == "cuda":
        torch.cuda.empty_cache()
    elif backend == "mps":
        torch.mps.empty_cache()
    elif backend == "xpu":
        torch.xpu.empty_cache()
    elif backend == "npu":
        torch.npu.empty_cache()


def set_device_seed(seed: int) -> None:
    """Set the seed for the current device."""
    backend = _get_active_backend_name()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if backend == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif backend == "xpu":
        torch.xpu.manual_seed_all(seed)
    elif backend == "npu":
        torch.npu.manual_seed_all(seed)
    elif backend == "mps":
        torch.mps.manual_seed(seed)
