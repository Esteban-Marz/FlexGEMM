from typing import *
import torch
import triton


def get_gpu_name() -> Optional[str]:
    """Return the GPU device name, or None if no CUDA GPU is available."""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name()
    except RuntimeError:
        return None


def get_platform_name():
    if torch.cuda.is_available():
        if getattr(torch.version, 'hip', None) is not None:
            return 'hip'
        return 'cuda'
    return 'unknown'
    

def get_num_sm():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties("cuda").multi_processor_count
    

def get_autotune_config(
    default: List[triton.Config] = None,
    platform: Dict[str, List[triton.Config]] = None,
    device: Dict[str, List[triton.Config]] = None,
) -> List[triton.Config]:
    """
    Get the autotune configuration for the current platform and device.

    Gracefully handles CPU-only environments (e.g. Modal memory snapshot phase)
    by skipping device-specific lookup when no GPU is available and falling
    through to platform or default configs. On CPU, the CUDA platform configs
    are used as a sensible fallback since autotune runs at kernel launch time
    when GPU is available.
    """
    gpu_available = torch.cuda.is_available()

    if device is not None and gpu_available:
        gpu_name = get_gpu_name()
        if gpu_name is not None:
            for key, value in device.items():
                if key.lower() in gpu_name.lower():
                    return value
    
    if platform is not None:
        platform_name = get_platform_name()
        # On CPU, fall back to CUDA configs (autotuning happens at kernel launch)
        if not gpu_available:
            platform_name = 'cuda'
        for key, value in platform.items():
            if key.lower() in platform_name.lower():
                return value
    
    if default is None:
        raise ValueError("No autotune configuration found for the current platform and device.")
    return default
