import os


def get_device():
    """Return the best available inference device for Ultralytics/PyTorch.

    Preference order: explicit ``FOOTBALL_DEVICE`` env override -> CUDA ->
    Intel GPU (XPU) -> Apple MPS -> CPU.

    Note on the return type: Ultralytics' ``select_device`` rejects the bare
    string ``"xpu"`` (it routes unknown strings through its CUDA parser). A
    ``torch.device`` object whose type is not ``"cuda"`` is passed through
    untouched, so we hand back ``torch.device("xpu")`` for Intel GPUs. CUDA
    still returns the ``"cuda:0"`` string so Ultralytics runs its normal
    device-selection / ``set_device`` logic.
    """
    import torch

    override = os.environ.get("FOOTBALL_DEVICE")
    if override:
        # A device index like "0" / "0,1" must stay a string for the CUDA path;
        # named accelerators go through as torch.device so they aren't parsed
        # as CUDA. "cuda"/"cpu"/"mps" strings are understood as-is.
        if override in {"cpu", "mps"} or override.startswith("cuda") or all(
            p.strip().isdigit() for p in override.split(",")
        ):
            return override
        return torch.device(override)

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
