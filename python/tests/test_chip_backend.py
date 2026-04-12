from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from ttt.int5_model import generate_int5, load_int5_model
from ttt.model import ModelConfig, TinyCharLm
from ttt.quantization import save_quantized_checkpoint
from ttt.runtime import Int5TiledRuntime, ReferenceSa2x2Device, VerilatedSa2x2Device


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_chip_backend_matches_reference_runtime_for_tiled_matmul() -> None:
    lhs = torch.tensor(
        [
            [1, -2, 3, 0],
            [2, 1, -1, 4],
            [0, 2, 3, -2],
        ],
        dtype=torch.int32,
    )
    rhs = torch.tensor(
        [
            [2, 1, -1, 0],
            [3, -2, 1, 4],
            [1, 0, 2, -3],
            [2, 1, -2, 1],
        ],
        dtype=torch.int32,
    )

    ref_runtime = Int5TiledRuntime(ReferenceSa2x2Device())
    chip_runtime = Int5TiledRuntime(VerilatedSa2x2Device())
    try:
        expected = ref_runtime.matmul(lhs, rhs, post_shift=2, relu=True, clamp_output_to_int5=True)
        got = chip_runtime.matmul(lhs, rhs, post_shift=2, relu=True, clamp_output_to_int5=True)
    finally:
        ref_runtime.close()
        chip_runtime.close()

    assert torch.equal(got, expected)


@pytest.mark.skipif(shutil.which("verilator") is None, reason="verilator not installed")
def test_chip_backend_matches_reference_generation(tmp_path: Path) -> None:
    torch.manual_seed(1)
    config = ModelConfig()
    model = TinyCharLm(config).eval()
    out_path = tmp_path / "model_int5.pt"
    save_quantized_checkpoint(out_path, model=model, model_config=config, activation_frac_bits=2)

    torch.manual_seed(2)
    ref_model = load_int5_model(out_path, backend="int5_ref")
    try:
        ref_out = generate_int5(ref_model, [1, 2, 3], max_new_tokens=4, temperature=0.8, top_k=8)
    finally:
        ref_model.close()

    torch.manual_seed(2)
    chip_model = load_int5_model(out_path, backend="chip_sim")
    try:
        chip_out = generate_int5(chip_model, [1, 2, 3], max_new_tokens=4, temperature=0.8, top_k=8)
    finally:
        chip_model.close()

    assert chip_out == ref_out
