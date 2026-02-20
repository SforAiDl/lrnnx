"""
Orchestrator script to run all benchmarks and save results to CSV.
"""

import csv
import os

import torch

# Model imports
from lrnnx.models.lti.lru import LRU
from lrnnx.models.lti.s5 import S5
from lrnnx.models.ltv.mamba import Mamba

from . import benchmark_inference as infer
from . import benchmark_training as train

STATE_DIM = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_models(model_dim: int, train_mode: bool = False):
    """Returns dict of model_name -> model_fn for a given model dimension."""
    mode = "train" if train_mode else "eval"
    return {
        "S5": lambda H=model_dim: getattr(
            S5(d_model=H, d_state=STATE_DIM, discretization="zoh").to(
                DEVICE
            ),
            mode,
        )(),
        "LRU": lambda H=model_dim: getattr(
            LRU(d_state=STATE_DIM, d_model=H).to(DEVICE), mode
        )(),
        "Mamba": lambda H=model_dim: getattr(
            Mamba(d_model=H, d_state=STATE_DIM).to(DEVICE), mode
        )(),
    }


def load_existing(filename: str) -> set:
    """Load existing (model, param_value) pairs from CSV."""
    existing = set()
    if os.path.exists(filename):
        with open(filename, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    existing.add((row[0], str(row[1])))
    return existing


def append_row(filename: str, header: list, row: list):
    """Append a single row to CSV, creating file with header if needed."""
    write_header = not os.path.exists(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def run_all_benchmarks(output_dir: str = "benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)

    seq_lengths = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    model_dims = [16, 32, 64, 128, 256]
    batch_sizes = [8, 16, 32, 64, 128]

    # === Inference Benchmarks ===
    print("\n=== Running Inference Benchmarks ===")

    # Sequence Length
    csv_file = f"{output_dir}/infer_seq_length.csv"
    header = ["model", "seq_length", "mean_ms", "min_ms", "max_ms"]
    existing = load_existing(csv_file)
    print("\n  [Inference] Sequence Length...")
    for name, model_fn in get_models(128).items():
        remaining = [L for L in seq_lengths if (name, str(L)) not in existing]
        if not remaining:
            print(f"    {name} (all skipped)")
            continue
        print(f"    {name} L={remaining}...")
        try:
            results = infer.benchmark_sequence_length(model_fn, remaining)
            for L, times in results.items():
                append_row(
                    csv_file,
                    header,
                    [name, L, sum(times) / len(times), min(times), max(times)],
                )
        except Exception as e:
            print(f"      FAILED: {e}")

    # Model Dimension
    csv_file = f"{output_dir}/infer_model_dim.csv"
    header = ["model", "model_dim", "mean_ms", "min_ms", "max_ms"]
    existing = load_existing(csv_file)
    print("\n  [Inference] Model Dimension...")
    for model_name in get_models(128).keys():
        remaining = [
            H for H in model_dims if (model_name, str(H)) not in existing
        ]
        if not remaining:
            print(f"    {model_name} (all skipped)")
            continue
        print(f"    {model_name} H={remaining}...")
        try:
            maker = lambda H, n=model_name: get_models(H)[n]
            results = infer.benchmark_model_dimension(maker, remaining)
            for H, times in results.items():
                append_row(
                    csv_file,
                    header,
                    [
                        model_name,
                        H,
                        sum(times) / len(times),
                        min(times),
                        max(times),
                    ],
                )
        except Exception as e:
            print(f"      FAILED: {e}")

    # Batch Size
    csv_file = f"{output_dir}/infer_batch_size.csv"
    header = ["model", "batch_size", "mean_ms", "min_ms", "max_ms"]
    existing = load_existing(csv_file)
    print("\n  [Inference] Batch Size...")
    for name, model_fn in get_models(128).items():
        remaining = [B for B in batch_sizes if (name, str(B)) not in existing]
        if not remaining:
            print(f"    {name} (all skipped)")
            continue
        print(f"    {name} B={remaining}...")
        try:
            results = infer.benchmark_batch_size(model_fn, remaining)
            for B, times in results.items():
                append_row(
                    csv_file,
                    header,
                    [name, B, sum(times) / len(times), min(times), max(times)],
                )
        except Exception as e:
            print(f"      FAILED: {e}")

    # === Training Benchmarks ===
    print("\n=== Running Training Benchmarks ===")

    # Sequence Length
    csv_file = f"{output_dir}/train_seq_length.csv"
    header = ["model", "seq_length", "mean_ms", "min_ms", "max_ms"]
    existing = load_existing(csv_file)
    print("\n  [Training] Sequence Length...")
    for name, model_fn in get_models(128, train_mode=True).items():
        remaining = [L for L in seq_lengths if (name, str(L)) not in existing]
        if not remaining:
            print(f"    {name} (all skipped)")
            continue
        print(f"    {name} L={remaining}...")
        try:
            results = train.benchmark_sequence_length(model_fn, remaining)
            for L, times in results.items():
                append_row(
                    csv_file,
                    header,
                    [name, L, sum(times) / len(times), min(times), max(times)],
                )
        except Exception as e:
            print(f"      FAILED: {e}")

    # Model Dimension
    csv_file = f"{output_dir}/train_model_dim.csv"
    header = ["model", "model_dim", "mean_ms", "min_ms", "max_ms"]
    existing = load_existing(csv_file)
    print("\n  [Training] Model Dimension...")
    for model_name in get_models(128, train_mode=True).keys():
        remaining = [
            H for H in model_dims if (model_name, str(H)) not in existing
        ]
        if not remaining:
            print(f"    {model_name} (all skipped)")
            continue
        print(f"    {model_name} H={remaining}...")
        try:
            maker = lambda H, n=model_name: get_models(H, train_mode=True)[n]
            results = train.benchmark_model_dimension(maker, remaining)
            for H, times in results.items():
                append_row(
                    csv_file,
                    header,
                    [
                        model_name,
                        H,
                        sum(times) / len(times),
                        min(times),
                        max(times),
                    ],
                )
        except Exception as e:
            print(f"      FAILED: {e}")

    # Batch Size
    csv_file = f"{output_dir}/train_batch_size.csv"
    header = ["model", "batch_size", "mean_ms", "min_ms", "max_ms"]
    existing = load_existing(csv_file)
    print("\n  [Training] Batch Size...")
    for name, model_fn in get_models(128, train_mode=True).items():
        remaining = [B for B in batch_sizes if (name, str(B)) not in existing]
        if not remaining:
            print(f"    {name} (all skipped)")
            continue
        print(f"    {name} B={remaining}...")
        try:
            results = train.benchmark_batch_size(model_fn, remaining)
            for B, times in results.items():
                append_row(
                    csv_file,
                    header,
                    [name, B, sum(times) / len(times), min(times), max(times)],
                )
        except Exception as e:
            print(f"      FAILED: {e}")

    print(f"\nAll benchmarks complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    run_all_benchmarks()
