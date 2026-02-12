"""
Tests for lrnnx.utils.generation: verifies that CUDA-graph and plain-loop
autoregressive generation produce identical outputs for every model, and
reports wall-clock speedups.

Run with:
    pytest tests/utils/test_generation.py -v -s
"""

import time

import pytest
import torch

from lrnnx.models.lti import LRU, S4
from lrnnx.models.lti import S5 as S5_LTI
from lrnnx.models.lti import CentaurusNeck, CentaurusPWNeck
from lrnnx.models.ltv import RGLRU
from lrnnx.models.ltv import S5 as S5_LTV
from lrnnx.models.ltv import S7, Mamba
from lrnnx.utils.generation import capture_graph, generate

B = 32
H = 32
NUM_STEPS = 128
N_RUNS = 30
N_WARMUP = 5
RTOL = 6e-4
ATOL = 2e-3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _s5_lti(disc: str):
    return S5_LTI(d_model=H, d_state=16, discretization=disc)


def _s5_ltv(disc: str, conj_sym: bool):
    return S5_LTV(
        d_model=H,
        d_state=16,
        discretization=disc,
        conj_sym=conj_sym,
        use_fast_path=True,  # test the non-fast path for better coverage
    )


def _s4():
    return S4(d_model=H, d_state=32, l_max=256, channels=1, transposed=False)


def _lru():
    return LRU(d_model=H, d_state=16)


def _centaurus_neck():
    return CentaurusNeck(d_model=H, d_state=8, sub_state_dim=4)


def _centaurus_pwneck():
    return CentaurusPWNeck(d_model=H, d_state=8, sub_state_dim=4)


def _mamba(disc: str):
    return Mamba(d_model=H, d_state=16, discretization=disc)


def _rglru():
    return RGLRU(d_model=H)


def _s7():
    return S7(d_model=H, d_state=16, J=1)


MODEL_CONFIGS = [
    # LTI - S5 as S5_LTI (only stable discretization is ZOH, so we test that one)
    # for others, use the LTV version.
    ("S5_LTI_zoh", _s5_lti, {"disc": "zoh"}),
    ("S5_LTV_zoh_conj", _s5_ltv, {"disc": "zoh", "conj_sym": True}),
    ("S5_LTV_zoh_no_conj", _s5_ltv, {"disc": "zoh", "conj_sym": False}),
    ("S5_LTV_bilinear_conj", _s5_ltv, {"disc": "bilinear", "conj_sym": True}),
    (
        "S5_LTV_bilinear_no_conj",
        _s5_ltv,
        {"disc": "bilinear", "conj_sym": False},
    ),
    ("S5_LTV_dirac_conj", _s5_ltv, {"disc": "dirac", "conj_sym": True}),
    ("S5_LTV_dirac_no_conj", _s5_ltv, {"disc": "dirac", "conj_sym": False}),
    # LTI - S4
    ("S4", _s4, {}),
    # LTI - LRU
    ("LRU", _lru, {}),
    # LTI - Centaurus
    ("CentaurusNeck", _centaurus_neck, {}),
    ("CentaurusPWNeck", _centaurus_pwneck, {}),
    # LTV - Mamba discretizations
    ("Mamba_mamba", _mamba, {"disc": "mamba"}),
    ("Mamba_zoh", _mamba, {"disc": "zoh"}),
    ("Mamba_bilinear", _mamba, {"disc": "bilinear"}),
    ("Mamba_dirac", _mamba, {"disc": "dirac"}),
    # LTV - RGLRU
    ("RGLRU", _rglru, {}),
    # LTV - S7
    ("S7", _s7, {}),
]

# Mamba configs to also test with event-mode integration timesteps
MAMBA_EVENT_CONFIGS = [
    ("Mamba_mamba_event", _mamba, {"disc": "mamba"}),
    ("Mamba_zoh_event", _mamba, {"disc": "zoh"}),
    ("Mamba_bilinear_event", _mamba, {"disc": "bilinear"}),
    ("Mamba_dirac_event", _mamba, {"disc": "dirac"}),
]


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


def _run_pair(model, x0, event_mode=False, integration_timesteps=None):
    """Run both CUDA-graph and for-loop paths; return outputs + timings."""
    cache = capture_graph(model, batch_size=B, H=H, event_mode=event_mode)

    # CUDA graph path
    graph_times = []
    for i in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_graph = generate(
            model,
            x0,
            NUM_STEPS,
            graph_cache=cache,
            integration_timesteps=integration_timesteps,
        )
        torch.cuda.synchronize()
        if i >= N_WARMUP:
            graph_times.append(time.perf_counter() - t0)

    # For-loop path
    loop_times = []
    for i in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_loop = generate(
            model,
            x0,
            NUM_STEPS,
            integration_timesteps=integration_timesteps,
        )
        torch.cuda.synchronize()
        if i >= N_WARMUP:
            loop_times.append(time.perf_counter() - t0)

    avg_graph = sum(graph_times) / len(graph_times)
    avg_loop = sum(loop_times) / len(loop_times)
    speedup = avg_loop / avg_graph if avg_graph > 0 else float("inf")

    return y_graph, y_loop, avg_graph, avg_loop, speedup


@pytest.mark.parametrize(
    "name, factory, kwargs",
    MODEL_CONFIGS,
    ids=[c[0] for c in MODEL_CONFIGS],
)
def test_generate_matches(name, factory, kwargs):
    """CUDA-graph and for-loop generation produce the same output."""
    model = factory(**kwargs).cuda().eval()
    x0 = torch.zeros(B, H, device="cuda")

    y_graph, y_loop, t_graph, t_loop, speedup = _run_pair(model, x0)

    assert y_graph.shape == (B, NUM_STEPS, H)
    assert y_loop.shape == (B, NUM_STEPS, H)
    assert not y_graph.isnan().any(), "CUDA-graph path produced NaN"
    assert not y_loop.isnan().any(), "For-loop path produced NaN"
    torch.testing.assert_close(y_graph, y_loop, rtol=RTOL, atol=ATOL)

    print(
        f"\n  {name:25s} | graph {t_graph*1e3:7.2f} ms | "
        f"loop {t_loop*1e3:7.2f} ms | speedup {speedup:.2f}x"
    )


@pytest.mark.parametrize(
    "name, factory, kwargs",
    MAMBA_EVENT_CONFIGS,
    ids=[c[0] for c in MAMBA_EVENT_CONFIGS],
)
def test_mamba_event_mode(name, factory, kwargs):
    """Mamba with integration_timesteps: graph vs loop match."""
    model = factory(**kwargs).cuda().eval()
    x0 = torch.randn(B, H, device="cuda") * 0.01
    dt = torch.rand(B, 1, device="cuda").abs() + 0.01  # positive dt

    y_graph, y_loop, t_graph, t_loop, speedup = _run_pair(
        model,
        x0,
        event_mode=True,
        integration_timesteps=dt,
    )

    assert y_graph.shape == (B, NUM_STEPS, H)
    assert y_loop.shape == (B, NUM_STEPS, H)
    assert not y_graph.isnan().any(), "CUDA-graph path produced NaN"
    assert not y_loop.isnan().any(), "For-loop path produced NaN"
    torch.testing.assert_close(y_graph, y_loop, rtol=RTOL, atol=ATOL)

    print(
        f"\n  {name:25s} | graph {t_graph*1e3:7.2f} ms | "
        f"loop {t_loop*1e3:7.2f} ms | speedup {speedup:.2f}x"
    )
