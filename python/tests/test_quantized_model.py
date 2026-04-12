from __future__ import annotations

from pathlib import Path

import torch

from ttt.int5_model import generate_int5, load_int5_model
from ttt.model import ModelConfig, TinyCharLm
from ttt.quantization import load_quantized_checkpoint, save_quantized_checkpoint


def test_quantized_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = ModelConfig()
    model = TinyCharLm(config).eval()
    out_path = tmp_path / "model_int5.pt"

    save_quantized_checkpoint(out_path, model=model, model_config=config, activation_frac_bits=2)
    checkpoint = load_quantized_checkpoint(out_path)

    assert checkpoint.activation_frac_bits == 2
    assert checkpoint.quantized_state_dict
    assert checkpoint.weight_frac_bits["lm_head.weight"] >= 0
    assert all(tensor.dtype == torch.int32 for tensor in checkpoint.quantized_state_dict.values())


def test_int5_reference_generation_is_deterministic(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = ModelConfig()
    model = TinyCharLm(config).eval()
    out_path = tmp_path / "model_int5.pt"
    save_quantized_checkpoint(out_path, model=model, model_config=config, activation_frac_bits=2)

    torch.manual_seed(0)
    int_model_a = load_int5_model(out_path, backend="int5_ref")
    try:
        out_a = generate_int5(int_model_a, start_token_ids=[1, 2, 3], max_new_tokens=6, temperature=0.9, top_k=8)
    finally:
        int_model_a.close()

    torch.manual_seed(0)
    int_model_b = load_int5_model(out_path, backend="int5_ref")
    try:
        out_b = generate_int5(int_model_b, start_token_ids=[1, 2, 3], max_new_tokens=6, temperature=0.9, top_k=8)
    finally:
        int_model_b.close()

    assert out_a == out_b
