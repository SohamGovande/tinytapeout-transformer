from __future__ import annotations

import math
from pathlib import Path

import torch

from .model import encode_text, sample_next_token
from .quantization import QuantizedCheckpoint, load_quantized_checkpoint
from .runtime import Int5TiledRuntime, ReferenceSa2x2Device, VerilatedSa2x2Device


class Int5CharLm:
    def __init__(
        self,
        checkpoint: QuantizedCheckpoint,
        runtime: Int5TiledRuntime,
    ):
        self.config = checkpoint.model_config
        self.activation_frac_bits = checkpoint.activation_frac_bits
        self.weight_frac_bits = checkpoint.weight_frac_bits
        self.state = {
            name: tensor.to(torch.int32)
            for name, tensor in checkpoint.quantized_state_dict.items()
        }
        self.runtime = runtime

        if self.config.head_dim <= 0 or (self.config.head_dim & (self.config.head_dim - 1)) != 0:
            raise ValueError("head_dim must be a positive power of two for shift-only scaling")

    def close(self) -> None:
        self.runtime.close()

    def _linear(
        self,
        x: torch.Tensor,
        weight_name: str,
        relu: bool = False,
        extra_shift: int = 0,
        clamp_output_to_int5: bool = True,
    ) -> torch.Tensor:
        weight = self.state[weight_name].to(torch.int32)
        shift = self.weight_frac_bits[weight_name] + extra_shift
        return self.runtime.matmul(
            x,
            weight.transpose(0, 1),
            post_shift=shift,
            relu=relu,
            clamp_output_to_int5=clamp_output_to_int5,
        )

    def _attention(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        prefix = f"blocks.{layer_idx}.attn"
        q = self._linear(x, f"{prefix}.q_proj.weight")
        k = self._linear(x, f"{prefix}.k_proj.weight")
        v = self._linear(x, f"{prefix}.v_proj.weight")

        steps = x.shape[0]
        q = q.view(steps, self.config.n_heads, self.config.head_dim)
        k = k.view(steps, self.config.n_heads, self.config.head_dim)
        v = v.view(steps, self.config.n_heads, self.config.head_dim)

        causal_mask = torch.tril(torch.ones((steps, steps), dtype=torch.bool))
        head_outputs: list[torch.Tensor] = []
        scale_shift = self.activation_frac_bits + int(math.log2(self.config.head_dim))

        for head_idx in range(self.config.n_heads):
            q_head = q[:, head_idx, :]
            k_head = k[:, head_idx, :]
            v_head = v[:, head_idx, :]

            scores = self.runtime.matmul(
                q_head,
                k_head.transpose(0, 1),
                post_shift=scale_shift,
                relu=True,
                clamp_output_to_int5=True,
                mask=causal_mask,
            )
            context = self.runtime.matmul(
                scores,
                v_head,
                post_shift=self.activation_frac_bits,
                relu=False,
                clamp_output_to_int5=True,
            )
            head_outputs.append(context)

        context = torch.cat(head_outputs, dim=1)
        return self._linear(context, f"{prefix}.out_proj.weight")

    def _mlp(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        prefix = f"blocks.{layer_idx}.mlp"
        hidden = self._linear(x, f"{prefix}.fc1.weight", relu=True)
        return self._linear(hidden, f"{prefix}.fc2.weight")

    def _forward_single(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(torch.long)
        positions = torch.arange(token_ids.numel(), dtype=torch.long)

        tok = self.state["token_embedding.weight"][token_ids]
        pos = self.state["position_embedding.weight"][positions]
        x = self.runtime.add(tok, pos, clamp_output_to_int5=True)

        for layer_idx in range(self.config.n_layers):
            attn_out = self._attention(x, layer_idx)
            x = self.runtime.add(x, attn_out, clamp_output_to_int5=True)
            mlp_out = self._mlp(x, layer_idx)
            x = self.runtime.add(x, mlp_out, clamp_output_to_int5=True)

        logits = self._linear(
            x,
            "lm_head.weight",
            clamp_output_to_int5=False,
        )
        return logits

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        outputs = [self._forward_single(row) for row in token_ids]
        return torch.stack(outputs, dim=0)

    def dequantize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.to(torch.float32) / float(1 << self.activation_frac_bits)


def load_int5_model(
    checkpoint_path: str | Path,
    backend: str = "int5_ref",
) -> Int5CharLm:
    checkpoint = load_quantized_checkpoint(checkpoint_path)
    if backend == "int5_ref":
        runtime = Int5TiledRuntime(ReferenceSa2x2Device())
    elif backend == "chip_sim":
        runtime = Int5TiledRuntime(VerilatedSa2x2Device())
    else:
        raise ValueError(f"unsupported int5 backend: {backend}")
    return Int5CharLm(checkpoint=checkpoint, runtime=runtime)


@torch.no_grad()
def generate_int5(
    model: Int5CharLm,
    start_token_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[int]:
    generated = list(start_token_ids)
    for _ in range(max_new_tokens):
        idx = torch.tensor([generated[-model.config.max_seq_len :]], dtype=torch.long)
        logits = model.forward(idx)
        float_logits = model.dequantize_logits(logits[0, -1, :])
        next_token = sample_next_token(
            float_logits,
            temperature=temperature,
            top_k=top_k,
        )
        generated.append(next_token)
    return generated


def encode_prompt(prompt: str) -> list[int]:
    return encode_text(prompt)
