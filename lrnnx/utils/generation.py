"""
CUDA-graph-accelerated step-by-step inference for LRNN models.

Inspired by https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/generation.py

Usage::

    cache = capture_graph(model, batch_size=4, H=64)
    y = generate(model, x0, num_steps=512, graph_cache=cache)  # CUDA-graph replay
    y = generate(model, x0, num_steps=512)                     # plain fallback
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lrnnx.models.lti.base import LTI_LRNN
    from lrnnx.models.ltv.base import LTV_LRNN

_STATE_KEYS = ("lrnn_state", "conv_state")


def _squeeze_out(y: Tensor) -> Tensor:
    return y.squeeze(1) if y.dim() == 3 else y


def _find_state_tensor(cache_dict: dict) -> Tensor:
    """Return the first Tensor value in a cache dict (fallback for state zeroing)."""
    for v in cache_dict.values():
        if isinstance(v, Tensor):
            return v
    raise ValueError("No Tensor found in inference_cache dict")


@dataclass
class CUDAGraphStepCache:
    """Holds a captured CUDA graph and the fixed-address buffers it operates on.

    Create instances via `capture_graph` - not directly.

    Attributes
    ----------
    graph : torch.cuda.CUDAGraph
        The captured graph.
    x_buf : Tensor
        Input buffer (batch, H) - write new data here before replay.
    y_buf : Tensor
        Output buffer (batch, H) - read result after replay.
    state_buf : Tensor
        Primary hidden-state buffer.
    mempool : int
        CUDA graph memory-pool handle (allows sharing across graphs).
    batch_size : int
        Batch size the graph was captured with.
    """

    graph: torch.cuda.CUDAGraph
    x_buf: Tensor
    y_buf: Tensor
    state_buf: Tensor
    mempool: int
    batch_size: int
    dt_buf: Optional[Tensor] = None
    _state_bufs: list = field(default_factory=list)
    _inference_cache: Optional[dict] = field(default=None, repr=False)


@torch.no_grad()
def capture_graph(
    model: LTI_LRNN | LTV_LRNN,
    batch_size: int,
    H: int,
    max_seqlen: int = 1,
    event_mode: bool = False,
    device: torch.device | str | None = None,
    n_warmups: int = 3,
) -> CUDAGraphStepCache:
    """Capture the model's single-step recurrence as a CUDA graph.

    Call this **once** (outside the hot loop) and pass the returned
    :class:`CUDAGraphStepCache` to :func:`generate` for zero-overhead
    replay.

    Parameters
    ----------
    model : LTI_LRNN | LTV_LRNN
        An lrnnx model on CUDA in eval mode.
    batch_size : int
        Batch size to capture for.  Every subsequent generate call
        must use the same batch size.
    H : int
        Model input/output dimension.
    max_seqlen : int
        Maximum sequence length (passed to allocate_inference_cache).
    event_mode : bool
        If True, capture with an integration_timesteps input buffer
        so that event-driven timesteps can be supplied at replay time.
    device : torch.device | str | None
        CUDA device.  Inferred from model parameters if None.
    n_warmups : int
        Number of warm-up iterations before capture (default 3).

    Returns
    -------
    CUDAGraphStepCache
        Opaque handle - pass it as graph_cache to :func:`generate`.
    """
    if device is None:
        device = next(model.parameters()).device

    # Free any stale graph memory before allocating new buffers
    gc.collect()
    torch.cuda.empty_cache()

    inference_cache = model.allocate_inference_cache(batch_size, max_seqlen)
    for k, v in inference_cache.items():
        if isinstance(v, Tensor):
            inference_cache[k] = v.to(device)

    x_buf = torch.zeros(batch_size, H, device=device, dtype=torch.float32)
    dt_buf: Tensor | None = None
    if event_mode:
        dt_buf = torch.ones(batch_size, 1, device=device, dtype=torch.float32)

    from lrnnx.models.ltv.base import LTV_LRNN as _LTV

    is_ltv = isinstance(model, _LTV)

    def _step(x_t):
        kwargs: Dict[str, Any] = {}
        if dt_buf is not None:
            kwargs["integration_timesteps"] = dt_buf
        x_in = x_t.unsqueeze(1) if is_ltv else x_t
        y, c = model.step(x_in, inference_cache, **kwargs)
        return _squeeze_out(y), c

    # Warm-up on a side stream (required before capture)
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            _step(x_buf)
    torch.cuda.current_stream(device).wait_stream(s)

    # Capture
    mempool = torch.cuda.graphs.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        y_buf, _ = _step(x_buf)

    # Collect state buffers for zeroing before each generation run
    state_bufs = [
        inference_cache[k]
        for k in _STATE_KEYS
        if k in inference_cache and isinstance(inference_cache[k], Tensor)
    ]

    return CUDAGraphStepCache(
        graph=graph,
        x_buf=x_buf,
        y_buf=y_buf,
        state_buf=state_bufs[0] if state_bufs else x_buf,
        mempool=mempool,
        batch_size=batch_size,
        dt_buf=dt_buf,
        _state_bufs=state_bufs,
        _inference_cache=inference_cache,
    )


def generate(
    model: LTI_LRNN | LTV_LRNN,
    x: Tensor,
    num_steps: int,
    graph_cache: CUDAGraphStepCache | None = None,
    integration_timesteps: Tensor | None = None,
) -> Tensor:
    """Autoregressive generation: feed each output back as the next input.

    When *graph_cache* is provided the pre-captured CUDA graph is
    **replayed** for every timestep - no re-capture, no extra overhead.
    When None, falls back to a plain Python loop.

    Parameters
    ----------
    model : LTI_LRNN | LTV_LRNN
        An lrnnx model on CUDA in eval mode.
    x : Tensor
        Seed input, shape (batch, H).
    num_steps : int
        Number of autoregressive steps to generate.
    graph_cache : CUDAGraphStepCache | None
        Pre-captured graph from :func:`capture_graph`.
    integration_timesteps : Tensor | None
        Integration timestep (batch, 1) for event-driven models,
        reused at every generated step.  Requires that graph_cache
        was captured with event_mode=True when using the CUDA-graph
        path.

    Returns
    -------
    Tensor
        Generated output sequence, shape (batch, num_steps, H).
    """
    if x.dim() != 2:
        raise ValueError(f"Expected x of shape (B, H), got {x.shape}")

    if graph_cache is not None:
        return _generate_with_cuda_graph(
            graph_cache, x, num_steps, integration_timesteps
        )
    return _generate_with_for_loop(model, x, num_steps, integration_timesteps)


def _generate_with_cuda_graph(
    cache: CUDAGraphStepCache,
    x: Tensor,
    num_steps: int,
    integration_timesteps: Tensor | None = None,
) -> Tensor:
    """Replay a captured CUDA graph for each autoregressive step."""
    batch, H = x.shape
    if batch != cache.batch_size:
        raise ValueError(
            f"Batch size {batch} != captured {cache.batch_size}. "
            f"Re-capture with capture_graph(model, batch_size={batch})."
        )

    # Reset all recurrent state tensors
    for buf in cache._state_bufs:
        buf.zero_()
    if (
        cache._inference_cache is not None
        and "seqlen_offset" in cache._inference_cache
    ):
        cache._inference_cache["seqlen_offset"] = 0

    outputs = torch.empty(
        batch, num_steps, H, device=x.device, dtype=torch.float32
    )
    cache.x_buf.copy_(x)
    if cache.dt_buf is not None and integration_timesteps is not None:
        cache.dt_buf.copy_(integration_timesteps)
    with torch.inference_mode():
        for t in range(num_steps):
            cache.graph.replay()
            outputs[:, t, :] = cache.y_buf
            cache.x_buf.copy_(cache.y_buf)
    return outputs


def _generate_with_for_loop(
    model: LTI_LRNN | LTV_LRNN,
    x: Tensor,
    num_steps: int,
    integration_timesteps: Tensor | None = None,
) -> Tensor:
    """Plain Python loop fallback (no CUDA graph)."""
    batch, H = x.shape
    device = x.device

    inference_cache = model.allocate_inference_cache(batch, num_steps)
    for k, v in inference_cache.items():
        if isinstance(v, Tensor):
            inference_cache[k] = v.to(device)

    from lrnnx.models.ltv.base import LTV_LRNN as _LTV

    is_ltv = isinstance(model, _LTV)
    outputs = torch.empty(
        batch, num_steps, H, device=device, dtype=torch.float32
    )
    x_t = x  # (B, H)
    with torch.inference_mode():
        for t in range(num_steps):
            kwargs: Dict[str, Any] = {}
            if integration_timesteps is not None:
                kwargs["integration_timesteps"] = integration_timesteps
            x_in = x_t.unsqueeze(1) if is_ltv else x_t
            y, inference_cache = model.step(x_in, inference_cache, **kwargs)
            y_flat = _squeeze_out(y)
            outputs[:, t, :] = y_flat
            x_t = y_flat
    return outputs
