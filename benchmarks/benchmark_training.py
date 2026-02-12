"""
Training benchmarking utilities for lrnnx models (forward + backward pass).
"""

import time
from typing import Callable, List

import torch


def _infer_H(model) -> int:
    """Infer the model / input dimension *H* from common attribute names."""
    for attr in ("H", "hid_dim", "d_model", "d_inner", "hidden_size"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError(
        f"Cannot infer H from {type(model).__name__}. "
        "Expected one of: H, hid_dim, d_model, d_inner, hidden_size"
    )


def _benchmark_once(
    model, x: torch.Tensor, warmup: int = 10, runs: int = 90
) -> float:
    """Run training benchmark: forward + backward. Returns mean time in ms."""
    model.train()
    device = next(model.parameters()).device
    x = x.to(device)

    for _ in range(warmup):
        x_in = x.clone().requires_grad_(True)
        y = model(x_in)
        loss = y.sum()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        x_in = x.clone().requires_grad_(True)
        t0 = time.perf_counter()
        y = model(x_in)
        loss = y.sum()
        loss.backward()
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
    """Benchmark training time vs sequence length.

    Parameters
    ----------
    model_fn : Callable
        Factory function returning a model on the target device.
    seq_lengths : List[int]
        Sequence lengths to benchmark.
    batch_size : int
        Batch size for training.
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

    results = {L: [] for L in seq_lengths}
    for L in seq_lengths:
        x = torch.randn(batch_size, L, H, device=dev)
        for _ in range(repeats):
            results[L].append(_benchmark_once(model, x))
    return results


def benchmark_model_dimension(
    model_fn_maker: Callable,
    model_dims: List[int] = [16, 32, 64, 128, 256],
    seq_len: int = 512,
    batch_size: int = 32,
    repeats: int = 5,
) -> dict:
    """Benchmark training time vs model dimension.

    Parameters
    ----------
    model_fn_maker : Callable
        Factory that takes *H* and returns a model factory function.
    model_dims : List[int]
        Model dimensions to benchmark.
    seq_len : int
        Sequence length for training.
    batch_size : int
        Batch size for training.
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
        x = torch.randn(batch_size, seq_len, H, device=dev)
        for _ in range(repeats):
            results[H].append(_benchmark_once(model, x))
    return results


def benchmark_batch_size(
    model_fn: Callable,
    batch_sizes: List[int] = [8, 16, 32, 64, 128],
    seq_len: int = 512,
    repeats: int = 5,
) -> dict:
    """Benchmark training time vs batch size.

    Parameters
    ----------
    model_fn : Callable
        Factory function returning a model on the target device.
    batch_sizes : List[int]
        Batch sizes to benchmark.
    seq_len : int
        Sequence length for training.
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
        x = torch.randn(B, seq_len, H, device=dev)
        for _ in range(repeats):
            results[B].append(_benchmark_once(model, x))
    return results
