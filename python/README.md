# TinyTapeout int5 Python tools

Run commands from the repository root.

## Install

```bash
uv sync
```

`chip_sim` uses Verilator, so install `verilator` before running that backend.

`pcb` talks to the stock Tiny Tapeout MicroPython firmware over the USB REPL using `pyserial`.

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
uv run python -m ttt.sample --backend pcb --pcb-port /dev/cu.usbmodemXXXX
```

## PCB backend

The `pcb` backend is meant for the Tiny Tapeout demo board plus breakout board running the stock TT firmware/SDK.

Before using it:

```bash
uv run python -m ttt.sample --backend pcb --help
```

Practical checklist:

1. Flash the stock Tiny Tapeout MicroPython firmware if your board is not already running it.
2. Plug the demo board into your Mac over USB-C.
3. Close Tiny Tapeout Commander, `screen`, `mpremote`, or any other serial client first.
4. Point `--pcb-port` at the board's serial device. On macOS this is usually a `/dev/cu.usbmodem...` path.
5. Run your sample or inference command with `--backend pcb`.

The backend:

- enables `tt_um_sohamgovande_transformer` on the board,
- switches the TT SDK into `ASIC_RP_CONTROL`,
- configures the bidirectional pins for this design,
- uploads a temporary helper into RAM through the REPL,
- manually clocks every command instead of relying on PWM auto-clocking.

Nothing is written to the board filesystem. Replugging or resetting the board clears the helper.

If you do not pass `--pcb-port`, the backend tries to auto-detect the board. You can also set:

```bash
export TTT_PCB_PORT=/dev/cu.usbmodemXXXX
export TTT_PCB_PROJECT=tt_um_sohamgovande_transformer
export TTT_PCB_BAUDRATE=115200
export TTT_PCB_TIMEOUT=1.0
```

The PCB path is designed for correctness first. It uses a tile-level RPC so it is much less chatty than per-register REPL calls, but it will still be much slower than `chip_sim`.
