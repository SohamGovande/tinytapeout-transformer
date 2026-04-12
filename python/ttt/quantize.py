from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import TinyCharLm, load_model_config
from .quantization import save_quantized_checkpoint

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_YAML = str(PACKAGE_ROOT / "model.yaml")
DEFAULT_CHECKPOINT = "model.pt"
DEFAULT_INT5_CHECKPOINT = "model_int5.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model-yaml", default=DEFAULT_MODEL_YAML)
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_INT5_CHECKPOINT)
    parser.add_argument("--activation-frac-bits", type=int, default=2)
    parser.add_argument("--max-weight-frac-bits", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    checkpoint = torch.load(Path(args.ckpt), map_location="cpu")
    float_state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    config = load_model_config(args.model_yaml)
    model = TinyCharLm(config)
    model.load_state_dict(float_state)
    model.eval()

    out_path = save_quantized_checkpoint(
        path=args.out,
        model=model,
        model_config=config,
        activation_frac_bits=args.activation_frac_bits,
        max_weight_frac_bits=args.max_weight_frac_bits,
    )
    print(f"[quantize] wrote {out_path}", flush=True)
    return out_path


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
