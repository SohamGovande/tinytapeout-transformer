# Tiny Tapeout 2x2 Systolic Primitive Engine

This repository is a Tiny Tapeout native scaffold for a very small signed 2x2 systolic-array primitive engine. It keeps the clean separation between compute RTL and verification that worked well in `../sa`, but strips away the scheduler, memory, harness, and larger multi-op infrastructure. The chip exposes two host-written source banks plus a dedicated result bank so software can stitch together tiny matmul and elementwise kernels.

## Repo layout

- `src/core/` contains the reusable processing element and fixed 2x2 array.
- `src/control/` contains the small controller that stores the `A`, `B`, and `C` banks, runs the array, and exposes the read mux.
- `src/tt_um_sohamgovande_sa2x2.sv` is the Tiny Tapeout top module.
- `test/` contains the cocotb testbench.
- `docs/info.md` documents the pin protocol and example usage.

## Quickstart

Install the Python test dependencies:

```sh
python3 -m pip install -r test/requirements.txt
```

Run the RTL testbench:

```sh
make test
```

Run lint:

```sh
make lint
```

Run the local Yosys synthesis sanity check:

```sh
make synth
```

## Protocol overview

The chip stores three signed 2x2 banks:

- `A[2][2]`: host-written source bank
- `B[2][2]`: host-written source bank
- `C[2][2]`: result and accumulation bank

All three banks are stored as signed 9-bit words. The systolic-array matmul path only consumes the low 4 bits of `A` and `B`, which keeps the internal core small. The host is responsible for requantizing values before reusing them as matrix-multiply operands.

The command interface supports:

- writing any `A` or `B` entry as a signed 9-bit value
- reading any `A`, `B`, or `C` entry
- `C = A x B`
- `C += A x B`
- `C[addr] = A[addr] + B[addr]`
- `C[addr] = relu(C[addr])`
- `C[addr] = C[addr] >>> shift`

Add and accumulate currently wrap in signed 9-bit arithmetic. Shift is arithmetic right shift on `C`.

## Synthesis note

`make synth` runs a generic Yosys flow and writes `build/synth.log`. On this machine, the current primitive-engine version synthesizes to roughly `2.4k` generic logic cells. That is directionally useful, but it is not a substitute for the real Tiny Tapeout Sky130 hardening flow.

## Tiny Tapeout flow

The root `Makefile` provides convenience targets for local testing and for Tiny Tapeout hardening tools:

- `make test`
- `make test-gl`
- `make lint`
- `make synth`
- `make tt-config`
- `make harden`
- `make warnings`
- `make png`

The Tiny Tapeout hardening targets expect the `tt-support-tools` repository to be present as the `tt/` submodule:

```sh
git submodule update --init --recursive
```
