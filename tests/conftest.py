import torch


def materialize_params(model: torch.nn.Module) -> torch.nn.Module:
    """Clone all parameters/buffers to detach from safetensors mmap storage.

    Workaround for bus error with safetensors 0.7.0 mmap on macOS aarch64.
    """
    for p in model.parameters():
        p.data = p.data.clone()
    for b in model.buffers():
        b.data = b.data.clone()
    return model
