from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .download_data import DATA_PATH, ensure_dataset
from .model import ModelConfig, STOI, TinyCharLm, count_model_weights, load_model_config, normalize_text

DEFAULT_MODEL_YAML = "model.yaml"
DEFAULT_CHECKPOINT = "model.pt"


@dataclass
class TrainArgs:
    model_yaml: str
    dataset: str
    ckpt: str
    batch_size: int
    lr: float
    weight_decay: float
    steps: int
    eval_every: int
    eval_batches: int
    log_every: int
    seed: int
    device: str


def parse_args(argv: list[str] | None = None) -> TrainArgs:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model-yaml", default=DEFAULT_MODEL_YAML)
    parser.add_argument("--dataset", default=str(DATA_PATH))
    parser.add_argument("--ckpt", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return TrainArgs(**vars(parser.parse_args(argv)))


def _load_token_tensor(dataset_path: Path) -> torch.Tensor:
    raw_text = dataset_path.read_text(encoding="utf-8")
    normalized = normalize_text(raw_text)
    token_ids = [STOI[ch] for ch in normalized] if normalized else []
    if len(token_ids) < 2:
        raise ValueError("dataset must contain at least two tokens after normalization")
    return torch.tensor(token_ids, dtype=torch.long)


def _split_train_val(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    split_idx = max(1, int(0.9 * tokens.numel()))
    split_idx = min(split_idx, tokens.numel() - 1)
    return tokens[:split_idx], tokens[split_idx:]


def _get_batch(
    token_stream: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = token_stream.numel() - block_size - 1
    if max_start < 0:
        raise ValueError("dataset is too short for the configured max_seq_len")
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([token_stream[int(s) : int(s) + block_size] for s in starts])
    y = torch.stack([token_stream[int(s) + 1 : int(s) + block_size + 1] for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def _estimate_losses(
    model: TinyCharLm,
    config: ModelConfig,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
) -> dict[str, float]:
    losses: dict[str, float] = {}
    model.eval()
    for split_name, split_tokens in (("train", train_tokens), ("val", val_tokens)):
        split_losses: list[float] = []
        for _ in range(eval_batches):
            xb, yb = _get_batch(split_tokens, batch_size, config.max_seq_len, device)
            _, loss = model(xb, yb)
            split_losses.append(float(loss.item()))
        losses[split_name] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def _save_latest_checkpoint(
    checkpoint_path: Path,
    model: TinyCharLm,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    args: TrainArgs,
    step: int,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "step": int(step),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_args": asdict(args),
    }
    torch.save(checkpoint, checkpoint_path)


def main(args: TrainArgs) -> Path:
    dataset_path = ensure_dataset(args.dataset)
    config = load_model_config(args.model_yaml)
    device = torch.device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokens = _load_token_tensor(dataset_path)
    train_tokens, val_tokens = _split_train_val(tokens)

    model = TinyCharLm(config).to(device)
    param_count = count_model_weights(model)
    print(f"[model] parameters={param_count:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_path = Path(args.ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        xb, yb = _get_batch(train_tokens, args.batch_size, config.max_seq_len, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            train_loss = float(loss.item())
            print(f"[train] step={step} loss={train_loss:.4f}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            losses = _estimate_losses(
                model=model,
                config=config,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                batch_size=args.batch_size,
                eval_batches=args.eval_batches,
                device=device,
            )
            print(
                f"[eval] step={step} train_loss={losses['train']:.4f} val_loss={losses['val']:.4f}",
                flush=True,
            )
            _save_latest_checkpoint(ckpt_path, model, optimizer, config, args, step)

    if not ckpt_path.exists():
        _save_latest_checkpoint(ckpt_path, model, optimizer, config, args, args.steps)

    return ckpt_path


def cli() -> None:
    main(parse_args())


if __name__ == "__main__":
    cli()
