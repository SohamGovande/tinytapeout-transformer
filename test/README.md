# Tiny Tapeout cocotb testbench

This testbench drives the Tiny Tapeout wrapper directly. The tests write 4-bit matrix entries into the controller, pulse `start`, wait for `done`, and then read back the stored 9-bit results one address at a time.

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
