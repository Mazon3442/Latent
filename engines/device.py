"""Where a worker runs its model: the device Latent asked for (LATENT_DEVICE), checked against this PyTorch.

Shared by the workers (each runs from its own venv, but this folder is on sys.path).
"""
import os


def build(torch):
    """What this PyTorch was built for: "cuda" | "rocm" | "xpu" | "cpu"."""
    v = torch.version
    if getattr(v, "hip", None):
        return "rocm"
    if getattr(v, "cuda", None):
        return "cuda"
    if getattr(v, "xpu", None):
        return "xpu"
    return "cpu"


def pick(torch):
    """(device, fallback). device is "cuda" | "xpu" | "cpu" (ROCm's PyTorch drives AMD GPUs as "cuda").
    fallback is None, or {"wanted", "build"} when a GPU was asked for but this PyTorch can't use it;
    Latent tells the user why and how to fix it."""
    wanted = os.environ.get("LATENT_DEVICE", "")
    if wanted in ("", "cuda") and torch.cuda.is_available():
        return "cuda", None
    if wanted == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu", None
    if wanted in ("cuda", "xpu"):
        return "cpu", {"wanted": wanted, "build": build(torch)}
    return "cpu", None
