"""
Inference benchmarking utilities using CUDA-graph-accelerated generation.

Usage
-----

    from lrnnx.models.lti import LRU

    def model_fn():
        return LRU(d_state=16, d_model=128).cuda().eval()

    # Benchmark varying sequence lengths
    results = benchmark_sequence_length(model_fn)

    # Benchmark varying model dimensions
    def model_fn_maker(H):
        return lambda: LRU(d_state=16, d_model=H).cuda().eval()
    results = benchmark_model_dimension(model_fn_maker)

    # Benchmark varying batch sizes
    results = benchmark_batch_size(model_fn)
"""

import time
from typing import Callable, List

import torch

from lrnnx.utils.generation import capture_graph, generate


def _infer_H(model) -> int:
    """Infer the model / input dimension *H* from common attribute names."""
    for attr in ("H", "hid_dim", "d_model", "d_inner", "hidden_size"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(
        f"Cannot infer H from {type(model).__name__}. "
        "Expected one of: H, hid_dim, d_model, d_inner, hidden_size"
    )


def _benchmark(run_fn, device, warmup=10, runs=90):
    """Warmup + timed runs.  Returns mean time in ms."""
    for _ in range(warmup):
        run_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def benchmark_sequence_length(
    model_fn: Callable,
    seq_lengths: List[int] = [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    batch_size: int = 32,
    repeats: int = 5,
) -> dict:
    """Benchmark CUDA-graph inference time vs sequence length.

    The graph is captured **once** for the given batch size and reused
    across all sequence lengths.

    Parameters
    ----------
    model_fn : Callable
        Factory function returning a model on CUDA in eval mode.
    seq_lengths : List[int]
        Sequence lengths (num_steps) to benchmark.
    batch_size : int
        Batch size for generation.
    repeats : int
        Number of timing repeats per sequence length.

    Returns
    -------
    dict
        Mapping from sequence length to list of timing results (ms).
    """
    model = model_fn()
    dev = next(model.parameters()).device
    H = _infer_H(model)
    cache = capture_graph(model, batch_size, H, device=dev)

    results = {L: [] for L in seq_lengths}
    for L in seq_lengths:
        # x0 is a seed token of shape (B, H)
        x0 = torch.randn(batch_size, H, device=dev)
        run_fn = lambda _x0=x0, _L=L: generate(
            model, _x0, num_steps=_L, graph_cache=cache
        )
        for _ in range(repeats):
            results[L].append(_benchmark(run_fn, dev))
    return results


def benchmark_model_dimension(
    model_fn_maker: Callable,
    model_dims: List[int] = [16, 32, 64, 128, 256],
    seq_len: int = 512,
    batch_size: int = 32,
    repeats: int = 5,
) -> dict:
    """Benchmark CUDA-graph inference time vs model dimension.

    A fresh model and graph are captured for each *H*.

    Parameters
    ----------
    model_fn_maker : Callable
        Factory that takes *H* and returns a model factory function.
        E.g. lambda H: lambda: LRU(d_state=16, d_model=H).cuda().eval()
    model_dims : List[int]
        Model dimensions to benchmark.
    seq_len : int
        Number of autoregressive steps to generate.
    batch_size : int
        Batch size for generation.
    repeats : int
        Number of timing repeats per model dimension.

    Returns
    -------
    dict
        Mapping from model dimension to list of timing results (ms).
    """
    results = {H: [] for H in model_dims}
    for H in model_dims:
        model = model_fn_maker(H)()
        dev = next(model.parameters()).device
        cache = capture_graph(model, batch_size, H, device=dev)
        # x0 is a seed token of shape (B, H)
        x0 = torch.randn(batch_size, H, device=dev)
        run_fn = lambda _x0=x0, _m=model, _c=cache: generate(
            _m, _x0, num_steps=seq_len, graph_cache=_c
        )
        for _ in range(repeats):
            results[H].append(_benchmark(run_fn, dev))
    return results


def benchmark_batch_size(
    model_fn: Callable,
    batch_sizes: List[int] = [8, 16, 32, 64, 128],
    seq_len: int = 512,
    repeats: int = 5,
) -> dict:
    """Benchmark CUDA-graph inference time vs batch size.

    A fresh graph is captured for each batch size (CUDA graphs are
    fixed-shape, so the graph must be re-captured when *B* changes).

    Parameters
    ----------
    model_fn : Callable
        Factory function returning a model on CUDA in eval mode.
    batch_sizes : List[int]
        Batch sizes to benchmark.
    seq_len : int
        Number of autoregressive steps to generate.
    repeats : int
        Number of timing repeats per batch size.

    Returns
    -------
    dict
        Mapping from batch size to list of timing results (ms).
    """
    results = {B: [] for B in batch_sizes}
    for B in batch_sizes:
        model = model_fn()
        dev = next(model.parameters()).device
        H = _infer_H(model)
        cache = capture_graph(model, B, H, device=dev)
        # x0 is a seed token of shape (B, H)
        x0 = torch.randn(B, H, device=dev)
        run_fn = lambda _x0=x0, _m=model, _c=cache: generate(
            _m, _x0, num_steps=seq_len, graph_cache=_c
        )
        for _ in range(repeats):
            results[B].append(_benchmark(run_fn, dev))
    return results
