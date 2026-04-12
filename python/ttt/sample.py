from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from .quantize import main as quantize_main
from .int5_model import generate_int5, load_int5_model
from .model import TinyCharLm, count_model_weights, decode_token_ids, encode_text, generate_float, load_model_config

DEFAULT_MODEL_YAML = "model.yaml"
DEFAULT_CHECKPOINT = "model.pt"
DEFAULT_INT5_CHECKPOINT = "model_int5.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model-yaml", default=DEFAULT_MODEL_YAML)
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--int5-ckpt", default=DEFAULT_INT5_CHECKPOINT)
    parser.add_argument("--backend", choices=("float", "int5_ref", "chip_sim"), default="float")
    parser.add_argument("--prompt", default="i")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--activation-frac-bits", type=int, default=2)
    return parser.parse_args(argv)


def _ensure_int5_checkpoint(args: argparse.Namespace) -> Path:
    out_path = Path(args.int5_ckpt)
    if out_path.exists():
        return out_path
    quantize_main(
        [
            "--model-yaml",
            args.model_yaml,
            "--ckpt",
            args.ckpt,
            "--out",
            args.int5_ckpt,
            "--activation-frac-bits",
            str(args.activation_frac_bits),
        ]
    )
    return out_path


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    prompt_ids = encode_text(args.prompt)

    if args.backend == "float":
        config = load_model_config(args.model_yaml)
        checkpoint = torch.load(Path(args.ckpt), map_location="cpu")
        model = TinyCharLm(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model = model.to(torch.device(args.device))
        param_count = count_model_weights(model)
        print(f"[model] parameters={param_count:,}", flush=True)
        generated_ids = generate_float(
            model=model,
            start_token_ids=prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    else:
        int5_ckpt = _ensure_int5_checkpoint(args)
        int_model = load_int5_model(int5_ckpt, backend=args.backend)
        try:
            generated_ids = generate_int5(
                model=int_model,
                start_token_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        finally:
            int_model.close()

    decoded = decode_token_ids(generated_ids)
    print(decoded)
    return decoded


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
