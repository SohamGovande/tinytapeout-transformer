from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import ModelConfig

INT5_MIN = -(1 << 4)
INT5_MAX = (1 << 4) - 1
INT11_MIN = -(1 << 10)
INT11_MAX = (1 << 10) - 1


def wrap_signed_scalar(value: int, width: int) -> int:
    mask = (1 << width) - 1
    raw = value & mask
    sign_bit = 1 << (width - 1)
    if raw & sign_bit:
        raw -= 1 << width
    return raw


def clamp_int5_tensor(values: torch.Tensor) -> torch.Tensor:
    return values.clamp(INT5_MIN, INT5_MAX).to(torch.int32)


def choose_frac_bits(values: torch.Tensor, max_frac_bits: int = 12) -> int:
    if values.numel() == 0:
        return max_frac_bits
    max_abs = float(values.detach().abs().max().item())
    if max_abs == 0.0:
        return max_frac_bits
    raw = math.floor(math.log2(INT5_MAX / max_abs))
    return max(0, min(max_frac_bits, int(raw)))


def quantize_tensor_int5(values: torch.Tensor, frac_bits: int) -> torch.Tensor:
    scale = float(1 << frac_bits)
    quantized = torch.round(values.detach().to(torch.float64) * scale)
    return quantized.clamp(INT5_MIN, INT5_MAX).to(torch.int8)


@dataclass
class QuantizedCheckpoint:
    model_config: ModelConfig
    activation_frac_bits: int
    quantized_state_dict: dict[str, torch.Tensor]
    weight_frac_bits: dict[str, int]


def quantize_state_dict_to_int5(
    state_dict: dict[str, torch.Tensor],
    activation_frac_bits: int = 2,
    max_weight_frac_bits: int = 12,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    quantized: dict[str, torch.Tensor] = {}
    weight_frac_bits: dict[str, int] = {}

    for name, value in state_dict.items():
        if not torch.is_floating_point(value):
            quantized[name] = value.to(torch.int8)
            continue

        if name in {"token_embedding.weight", "position_embedding.weight"}:
            frac_bits = activation_frac_bits
        else:
            frac_bits = choose_frac_bits(value, max_frac_bits=max_weight_frac_bits)
            weight_frac_bits[name] = frac_bits

        quantized[name] = quantize_tensor_int5(value, frac_bits)

    return quantized, weight_frac_bits


def save_quantized_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    model_config: ModelConfig,
    activation_frac_bits: int = 2,
    max_weight_frac_bits: int = 12,
) -> Path:
    quantized_state_dict, weight_frac_bits = quantize_state_dict_to_int5(
        model.state_dict(),
        activation_frac_bits=activation_frac_bits,
        max_weight_frac_bits=max_weight_frac_bits,
    )
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config.__dict__,
            "activation_frac_bits": activation_frac_bits,
            "quantized_state_dict": quantized_state_dict,
            "weight_frac_bits": weight_frac_bits,
        },
        out_path,
    )
    return out_path


def load_quantized_checkpoint(path: str | Path) -> QuantizedCheckpoint:
    payload = torch.load(Path(path), map_location="cpu")
    return QuantizedCheckpoint(
        model_config=ModelConfig(**payload["model_config"]),
        activation_frac_bits=int(payload["activation_frac_bits"]),
        quantized_state_dict={
            name: tensor.to(torch.int32)
            for name, tensor in payload["quantized_state_dict"].items()
        },
        weight_frac_bits={name: int(bits) for name, bits in payload["weight_frac_bits"].items()},
    )
