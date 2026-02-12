"""
Unit tests for language model architecture.
"""

import os
import tempfile

import pytest
import torch

from lrnnx.architectures.language_model import LRNNLMHeadModel

LRNN_CONFIGS = [
    ("LRU", {"LRU": {}}),
    ("S4", {"S4": {"transposed": False}}),
    ("S4D", {"S4D": {"disc": "zoh", "transposed": False}}),
    ("S5", {"S5": {"discretization": "zoh"}}),
    ("Centaurus", {"Centaurus": {"sub_state_dim": 8}}),
    ("Mamba", {"Mamba": {"discretization": "mamba"}}),
    ("RGLRU", {"RGLRU": {}}),
    ("S7", {"S7": {}}),
]


@pytest.fixture(autouse=True)
def device_based(request):
    """Test based on device."""
    if "device" in request.fixturenames:
        device = request.getfixturevalue("device")
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if device == "cpu" and torch.cuda.is_available():
            pytest.xfail(
                reason="ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)"
            )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "mixer_type,mixer_kwargs",
    LRNN_CONFIGS,
    ids=[config[0] for config in LRNN_CONFIGS],
)
def test_language_model(device, mixer_type, mixer_kwargs):
    """Test pure LRNN language model with comprehensive forward passes."""

    batch_size = 32
    seq_len = 128
    dtype = torch.float32

    mixer_types = [mixer_type] * 12

    # use fused_add_norm only on CUDA
    fused_add_norm = device == "cuda"
    residual_in_fp32 = device == "cuda"

    model = LRNNLMHeadModel(
        d_model=256,
        d_state=128,
        n_layer=12,
        vocab_size=1000,
        mixer_types=mixer_types,
        mixer_kwargs=mixer_kwargs,
        rms_norm=True,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
        pad_vocab_size_multiple=16,
    )
    model = model.to(device=device)

    # test forward pass with random input
    torch.manual_seed(2357)
    input_ids = torch.randint(
        0, 1000, (batch_size, seq_len), device=device, dtype=torch.long
    )

    # forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits

    # use actual vocab size from model (which may be padded)
    actual_vocab_size = model.lm_head.out_features
    assert logits.shape == (batch_size, seq_len, actual_vocab_size)

    # test inference cache allocation
    cache = model.allocate_inference_cache(batch_size, seq_len)
    assert len(cache) == 12  # n_layer = 12

    # test autoregressive inference pattern
    prompt_len = seq_len // 2

    # full forward pass
    full_logits = model(input_ids).logits

    # step by step inference using cache
    with torch.no_grad():
        # process prompt all at once
        prompt_output = model(input_ids[:, :prompt_len])
        step_logits = [prompt_output.logits]

        # continue token by token for remaining sequence
        for i in range(seq_len - prompt_len):
            if prompt_len + i < seq_len:
                current_input = input_ids[
                    :, prompt_len + i : prompt_len + i + 1
                ]
                output = model(current_input)
                step_logits.append(output.logits)

    # compare consistency
    step_prompt_logits = step_logits[0]
    prompt_full_logits = full_logits[:, :prompt_len, :]
    assert torch.allclose(step_prompt_logits, prompt_full_logits, atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "mixer_type,mixer_kwargs",
    LRNN_CONFIGS,
    ids=[config[0] for config in LRNN_CONFIGS],
)
def test_save_load(device, mixer_type, mixer_kwargs):
    """Test saving and loading different LRNN models."""
    fused_add_norm = device == "cuda"
    residual_in_fp32 = device == "cuda"

    mixer_types = [mixer_type] * 12

    model = LRNNLMHeadModel(
        d_model=256,
        d_state=128,
        n_layer=12,
        vocab_size=32000,
        mixer_types=mixer_types,
        mixer_kwargs=mixer_kwargs,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    model.to(device)

    # create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # save model
        model.save_pretrained(temp_dir)

        # check files exist
        config_path = os.path.join(temp_dir, "config.json")
        model_path = os.path.join(temp_dir, "pytorch_model.bin")

        assert os.path.exists(config_path)
        assert os.path.exists(model_path)

        # load model
        loaded_model = LRNNLMHeadModel.from_pretrained(
            temp_dir, mixer_kwargs=mixer_kwargs
        )
        loaded_model.to(device)

        # test that loaded model works
        input_ids = torch.randint(0, 32000, (1, 5), device=device)
        with torch.no_grad():
            original_output = model(input_ids)
            loaded_output = loaded_model(input_ids)

        # check outputs are correct
        assert torch.allclose(
            original_output.logits, loaded_output.logits, atol=1e-4
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixed_model(device):
    """Test creating a model with different mixer types per layer."""
    fused_add_norm = device == "cuda"
    residual_in_fp32 = device == "cuda"

    mixer_types = ["LRU", "S5", "attn", "LRU", "S5", "attn"]
    mixer_kwargs = {
        "LRU": {},
        "S5": {"discretization": "zoh"},
        "attn": {
            "num_heads": 8,
            "causal": True,
            "qkv_proj_bias": True,
            "out_proj_bias": True,
        },
    }

    model = LRNNLMHeadModel(
        d_model=256,
        d_state=128,
        n_layer=6,
        vocab_size=1000,
        mixer_types=mixer_types,
        mixer_kwargs=mixer_kwargs,
        d_intermediate=512,
        rms_norm=True,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
    )
    model.to(device)

    # test forward pass
    batch_size = 32
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits

    assert logits.shape == (batch_size, seq_len, model.lm_head.out_features)

    # verify layer types match specification
    expected_types = ["LRU", "S5", "MHA", "LRU", "S5", "MHA"]
    for i, layer in enumerate(model.backbone.layers):
        mixer_type = type(layer.mixer).__name__
        assert (
            mixer_type == expected_types[i]
        ), f"Layer {i} should be {expected_types[i]}, got {mixer_type}"

    # test inference cache allocation
    cache = model.allocate_inference_cache(batch_size, seq_len)
    assert len(cache) == 6  # n_layer = 6


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "mixer_type,mixer_kwargs",
    LRNN_CONFIGS,
    ids=[config[0] for config in LRNN_CONFIGS],
)
def test_step_method(device, mixer_type, mixer_kwargs):
    """Test LRNN language model step method for autoregressive generation."""
    batch_size = 32
    seq_len = 128
    dtype = torch.float32

    fused_add_norm = device == "cuda"
    residual_in_fp32 = device == "cuda"

    mixer_types = [mixer_type] * 12

    model = LRNNLMHeadModel(
        d_model=256,
        d_state=128,
        n_layer=12,
        vocab_size=1000,
        mixer_types=mixer_types,
        mixer_kwargs=mixer_kwargs,
        rms_norm=True,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
        pad_vocab_size_multiple=16,
    )
    model = model.to(device=device)
    model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(
        0, 1000, (batch_size, seq_len), device=device, dtype=torch.long
    )

    with torch.no_grad():
        # get full sequence output using regular forward pass
        full_output = model(input_ids)
        full_logits = full_output.logits

        # use step
        caches = model.allocate_inference_cache(
            batch_size, max_seqlen=seq_len, dtype=dtype
        )

        step_logits = []
        for t in range(seq_len):
            token = input_ids[:, t : t + 1]  # (B, 1)
            step_output = model.step(token, caches)
            step_logits.append(step_output.logits)

        # stack step outputs
        step_logits = torch.cat(step_logits, dim=1)  # (B, L, vocab_size)

        assert step_logits.shape == full_logits.shape
        assert torch.allclose(
            step_logits, full_logits, rtol=0, atol=1e-4
        ), "Step-wise and full forward outputs should match"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixed_step_method(device):
    """Test mixed model with step method."""
    batch_size = 32
    seq_len = 128
    dtype = torch.float32

    mixer_types = ["LRU", "S5", "LRU", "S5"]
    mixer_kwargs = {
        "LRU": {},
        "S5": {"discretization": "zoh"},
    }

    fused_add_norm = device == "cuda"
    residual_in_fp32 = device == "cuda"

    model = LRNNLMHeadModel(
        d_model=128,
        d_state=64,
        n_layer=4,
        vocab_size=1000,
        mixer_types=mixer_types,
        mixer_kwargs=mixer_kwargs,
        rms_norm=True,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
        pad_vocab_size_multiple=16,
    )
    model = model.to(device=device)
    model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(
        0, 1000, (batch_size, seq_len), device=device, dtype=torch.long
    )

    with torch.no_grad():
        # get full sequence output using regular forward pass
        full_output = model(input_ids)
        full_logits = full_output.logits

        # use step
        caches = model.allocate_inference_cache(
            batch_size, max_seqlen=seq_len, dtype=dtype
        )

        step_logits = []
        for t in range(seq_len):
            token = input_ids[:, t : t + 1]  # (B, 1)
            step_output = model.step(token, caches)
            step_logits.append(step_output.logits)

        # stack step outputs
        step_logits = torch.cat(step_logits, dim=1)  # (B, L, vocab_size)

        assert step_logits.shape == full_logits.shape
        assert torch.allclose(
            step_logits, full_logits, rtol=0, atol=1e-4
        ), "Step-wise and full forward outputs should match"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mamba_with_event_mode(device):
    """Test Mamba model with event-based processing using integration_timesteps."""
    batch_size = 8
    seq_len = 64
    dtype = torch.float32

    fused_add_norm = device == "cuda"
    residual_in_fp32 = device == "cuda"

    mixer_types = ["Mamba"] * 4
    mixer_kwargs = {
        "Mamba": {"discretization": "mamba"},
    }

    model = LRNNLMHeadModel(
        d_model=128,
        d_state=64,
        n_layer=4,
        vocab_size=1000,
        mixer_types=mixer_types,
        mixer_kwargs=mixer_kwargs,
        rms_norm=True,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
        pad_vocab_size_multiple=16,
    )
    model = model.to(device=device)
    model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(
        0, 1000, (batch_size, seq_len), device=device, dtype=torch.long
    )

    integration_timesteps = (
        torch.rand(batch_size, seq_len, device=device, dtype=dtype) * 0.1
        + 0.01
    )

    with torch.no_grad():
        output_event = model(
            input_ids, integration_timesteps=integration_timesteps
        )
        logits_event = output_event.logits

        output_standard = model(input_ids)
        logits_standard = output_standard.logits

        assert logits_event.shape == (
            batch_size,
            seq_len,
            model.lm_head.out_features,
        )
        assert logits_standard.shape == (
            batch_size,
            seq_len,
            model.lm_head.out_features,
        )

        assert not torch.allclose(logits_event, logits_standard, atol=1e-3)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mamba_step_with_event_mode(device):
    """Test Mamba step method with event-based processing."""
    batch_size = 8
    seq_len = 32
    dtype = torch.float32

    fused_add_norm = device == "cuda"
    residual_in_fp32 = device == "cuda"

    mixer_types = ["Mamba"] * 4
    mixer_kwargs = {
        "Mamba": {"discretization": "mamba"},
    }

    model = LRNNLMHeadModel(
        d_model=128,
        d_state=64,
        n_layer=4,
        vocab_size=1000,
        mixer_types=mixer_types,
        mixer_kwargs=mixer_kwargs,
        rms_norm=True,
        residual_in_fp32=residual_in_fp32,
        fused_add_norm=fused_add_norm,
        pad_vocab_size_multiple=16,
    )
    model = model.to(device=device)
    model.eval()

    torch.manual_seed(42)
    input_ids = torch.randint(
        0, 1000, (batch_size, seq_len), device=device, dtype=torch.long
    )

    integration_timesteps = (
        torch.rand(batch_size, seq_len, device=device, dtype=dtype) * 0.1
        + 0.01
    )

    with torch.no_grad():
        # full forward pass with event mode
        full_output = model(
            input_ids, integration_timesteps=integration_timesteps
        )
        full_logits = full_output.logits

        # step-by-step inference with event mode
        caches = model.allocate_inference_cache(
            batch_size, max_seqlen=seq_len, dtype=dtype
        )

        step_logits = []
        for t in range(seq_len):
            token = input_ids[:, t : t + 1]  # (B, 1)
            dt = integration_timesteps[:, t : t + 1]  # (B, 1)
            step_output = model.step(token, caches, integration_timesteps=dt)
            step_logits.append(step_output.logits)

        step_logits = torch.cat(step_logits, dim=1)  # (B, L, vocab_size)

        assert step_logits.shape == full_logits.shape

        assert torch.allclose(
            step_logits, full_logits, rtol=0, atol=1e-3
        ), "Step-wise and full forward outputs should be close"
