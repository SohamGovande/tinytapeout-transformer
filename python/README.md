# TinyTapeout int5 Python tools

Run commands from this directory:

```bash
cd python
```

## Install

```bash
uv sync
```

`chip_sim` uses Verilator, so install `verilator` before running that backend.

## Train

```bash
uv run python -m ttt.train --steps 200
```

## Quantize to int5

```bash
uv run python -m ttt.quantize
```

## Sample

```bash
uv run python -m ttt.sample --backend float
uv run python -m ttt.sample --backend int5_ref
uv run python -m ttt.sample --backend chip_sim
```
