from __future__ import annotations

from pathlib import Path

import torch

from ttt.quantize import main as quantize_main
from ttt.sample import main as sample_main
from ttt.train import main as train_main, parse_args as parse_train_args


def test_train_quantize_and_sample_smoke(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("hello world " * 32, encoding="utf-8")

    ckpt_path = tmp_path / "model.pt"
    int5_path = tmp_path / "model_int5.pt"

    train_args = parse_train_args(
        [
            "--dataset",
            str(dataset_path),
            "--ckpt",
            str(ckpt_path),
            "--steps",
            "2",
            "--eval-every",
            "1",
            "--eval-batches",
            "1",
            "--log-every",
            "1",
            "--batch-size",
            "4",
        ]
    )
    train_main(train_args)
    assert ckpt_path.exists()

    quantize_main(
        [
            "--ckpt",
            str(ckpt_path),
            "--out",
            str(int5_path),
        ]
    )
    assert int5_path.exists()

    torch.manual_seed(0)
    decoded = sample_main(
        [
            "--ckpt",
            str(ckpt_path),
            "--int5-ckpt",
            str(int5_path),
            "--backend",
            "int5_ref",
            "--prompt",
            "he",
            "--max-new-tokens",
            "3",
        ]
    )
    assert isinstance(decoded, str)
    assert len(decoded) >= 2
