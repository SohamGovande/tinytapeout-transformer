# Tiny Tapeout cocotb testbench

This testbench drives the Tiny Tapeout wrapper directly. The tests write signed 5-bit matrix entries into the `A` and `B` banks, launch operations through the execute command, and read back signed results over the 9-bit Tiny Tapeout read bus. The `C` bank is 11 bits wide, so `C` reads are reconstructed from two bus chunks.

## Setup

Install the Python dependencies:

```sh
python3 -m pip install -r requirements.txt
```

By default the testbench uses Verilator because this repo is written in SystemVerilog and the root workflow already relies on Verilator for linting.

## Run RTL simulation

```sh
make -B
```

## Run gate-level simulation

After hardening, copy the generated gate-level netlist to `test/gate_level_netlist.v`, then run:

```sh
make -B GATES=yes
```

## Waveforms

The testbench dumps `tb.fst`, which can be viewed with:

```sh
gtkwave tb.fst
```
