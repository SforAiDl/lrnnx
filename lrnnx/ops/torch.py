from functools import partial
from typing import Callable

import torch


def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool):
    """
    Wrapper for Automatic Mixed Precision (AMP) decorators to handle deprecation.

    PyTorch deprecated ``torch.cuda.amp`` in favor of ``torch.amp``. This decorator
    ensures backward compatibility by injecting the ``device_type="cuda"`` keyword
    argument into the new decorator if the deprecated version is no longer used.

    Args:
        dec (Callable): The original AMP decorator function (e.g., ``custom_fwd`` or ``custom_bwd``).
        cuda_amp_deprecated (bool): A flag indicating whether the ``torch.cuda.amp`` module is deprecated.

    Returns:
        Callable: The wrapped decorator function.
    """

    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs["device_type"] = "cuda"
        return dec(*args, **kwargs)

    return decorator


if hasattr(torch.amp, "custom_fwd"):  # type: ignore[attr-defined]
    deprecated = True
    from torch.amp import custom_bwd, custom_fwd  # type: ignore[attr-defined]
else:
    deprecated = False
    from torch.cuda.amp import custom_bwd, custom_fwd

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)