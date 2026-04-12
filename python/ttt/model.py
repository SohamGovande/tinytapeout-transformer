from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

VOCAB_CHARS = " abcdefghijklmnopqrstuvwxyz"

STOI = {ch: idx for idx, ch in enumerate(VOCAB_CHARS)}
ITOS = {idx: ch for idx, ch in enumerate(VOCAB_CHARS)}


def normalize_text(text: str) -> str:
    lowered = text.lower()
    filtered = "".join(ch if ch in VOCAB_CHARS else " " for ch in lowered)
    return " ".join(filtered.split())


def encode_text(text: str) -> list[int]:
    normalized = normalize_text(text)
    if not normalized:
        return [STOI[" "]]
    return [STOI[ch] for ch in normalized]


def decode_token_ids(token_ids: list[int] | torch.Tensor) -> str:
    if isinstance(token_ids, torch.Tensor):
        flat = token_ids.detach().reshape(-1).tolist()
    else:
        flat = token_ids
    return "".join(ITOS[int(idx)] for idx in flat)


@dataclass
class ModelConfig:
    vocab_size: int = 27
    max_seq_len: int = 8
    d_model: int = 8
    n_layers: int = 1
    n_heads: int = 1
    head_dim: int = 8
    mlp_hidden_dim: int = 16

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")


def load_model_config(model_yaml: str | Path) -> ModelConfig:
    payload = yaml.safe_load(Path(model_yaml).read_text(encoding="utf-8")) or {}
    return ModelConfig(**payload)


def count_model_weights(model_or_state: nn.Module | dict[str, torch.Tensor]) -> int:
    if isinstance(model_or_state, nn.Module):
        return sum(int(param.numel()) for param in model_or_state.parameters())
    return sum(int(tensor.numel()) for tensor in model_or_state.values())


class ReluSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        q = self.q_proj(x).view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / float(self.head_dim)
        causal_mask = torch.tril(torch.ones(steps, steps, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, 0.0)
        scores = F.relu(scores)

        context = torch.matmul(scores, v)
        context = context.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.out_proj(context)


class ReluMlp(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.mlp_hidden_dim, bias=False)
        self.fc2 = nn.Linear(config.mlp_hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = ReluSelfAttention(config)
        self.mlp = ReluMlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class TinyCharLm(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, steps = token_ids.shape
        positions = torch.arange(steps, device=token_ids.device, dtype=torch.long)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss


def sample_next_token(
    next_logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> int:
    logits = next_logits / temperature
    if top_k is not None:
        k = min(top_k, logits.numel())
        top_values, _ = torch.topk(logits, k=k)
        cutoff = top_values[-1]
        neg_inf = torch.full_like(logits, float("-inf"))
        logits = torch.where(logits < cutoff, neg_inf, logits)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_float(
    model: TinyCharLm,
    start_token_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> list[int]:
    generated = list(start_token_ids)
    device = next(model.parameters()).device

    for _ in range(max_new_tokens):
        idx = torch.tensor(
            [generated[-model.config.max_seq_len :]],
            dtype=torch.long,
            device=device,
        )
        logits, _ = model(idx)
        next_token = sample_next_token(
            logits[0, -1, :],
            temperature=temperature,
            top_k=top_k,
        )
        generated.append(next_token)

    return generated
