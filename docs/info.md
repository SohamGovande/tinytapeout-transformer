## How it works

This design wraps a fixed signed 2x2 systolic array in a tiny primitive-engine interface that is easy to drive from a Tiny Tapeout board or from simulation.

The internal datapath has:

- a 2x2 bank of multiply-accumulate processing elements
- a host-written 2x2 source bank `A`
- a host-written 2x2 source bank `B`
- a 2x2 result and accumulation bank `C`
- a small addressable read mux for returning one stored word at a time

The `A` and `B` banks store signed 5-bit operands. The `C` bank stores signed 11-bit results, which is wide enough to hold the exact sum of two signed 5-bit products.

Although the external interface is simple, the controller still feeds the internal array using the correct skewed systolic schedule. That keeps the core architecture aligned with the larger `../sa` project while making this Tiny Tapeout design much smaller.

## Pin protocol

### Inputs

| Pin | Direction | Meaning |
| --- | --- | --- |
| `ui_in[7:0]` | input | command payload; write data, read-chunk select, or an op/immediate field |
| `uio_in[0]` | input | `cmd_stb`, pulse high for one cycle to issue a command |
| `uio_in[2:1]` | input | `cmd` |
| `uio_in[4:3]` | input | row-major element address |
| `uio_in[6]` | input | auxiliary write bit; unused in the default 5-bit operand mode |
| `uio_in[7]` | input | ignored; this pin is driven as an output |
| `uio_in[5]` | input | ignored; this pin is driven as an output |

### Command encoding

| `cmd` | Operation |
| --- | --- |
| `2'b00` | write `A[addr] <= ui_in[4:0]` |
| `2'b01` | write `B[addr] <= ui_in[4:0]` |
| `2'b10` | execute an operation encoded in `ui_in` |
| `2'b11` | select which bank, address, and read chunk appear on the output bus |

### Address map

| `addr` | Matrix entry |
| --- | --- |
| `0` | `[0][0]` |
| `1` | `[0][1]` |
| `2` | `[1][0]` |
| `3` | `[1][1]` |

### Read bank select

For `cmd=2'b11`, the read bank select is carried in `ui_in[1:0]` and the chunk index is carried in `ui_in[7:2]`:

| `ui_in[1:0]` | Read bank |
| --- | --- |
| `0` | `A` |
| `1` | `B` |
| `2` | `C` |

Chunk behavior:

- `A` and `B` only need chunk `0`.
- `C` chunk `0` returns bits `[8:0]`.
- `C` chunk `1` returns bits `[10:9]` in the low bits of the 9-bit read bus.

### Execute payload format

For `cmd=2'b10`, `ui_in` is interpreted as:

```text
ui_in[2:0] = op
ui_in[3]   = reserved
ui_in[7:4] = shift amount for OP_EW_SHIFT
```

Supported ops:

| `ui_in[2:0]` | Operation |
| --- | --- |
| `3'b000` | `C = A x B` |
| `3'b001` | `C = C + (A x B)` |
| `3'b010` | `C[addr] = A[addr] + B[addr]` |
| `3'b011` | `C[addr] = relu(C[addr])` |
| `3'b100` | `C[addr] = C[addr] >>> ui_in[7:4]` |

### Outputs

| Pin | Direction | Meaning |
| --- | --- | --- |
| `uo_out[7:0]` | output | low 8 bits of the selected signed readback chunk |
| `uio_out[7]` | output | high bit of the selected signed readback chunk |
| `uio_out[6]` | output | zero |
| `uio_out[5]` | output | `busy` |
| `uio_out[4:0]` | output | zero |
| `uio_oe[7:0]` | output-enable | `8'b1010_0000` |

Interpret each returned chunk as `{uio_out[7], uo_out[7:0]}` and reconstruct 11-bit `C` values from two chunks.

## Timing for one matmul

From the external interface, the flow is:

1. Write four entries of `A`.
2. Write four entries of `B`.
3. Issue `cmd=2'b10` with `ui_in[2:0]=3'b000`.
4. Wait for `busy=0`.
5. Use `cmd=2'b11` and `addr=0..3` to read back `C`.

Internally, after the execute command, the controller steps through:

| Cycle | Internal state | Action |
| --- | --- | --- |
| 1 | `CLEAR` | clear all PE accumulators |
| 2 | `FEED0` | inject the first skewed wavefront |
| 3 | `FEED1` | inject the second skewed wavefront |
| 4 | `FLUSH0` | inject the final skewed values needed for the bottom-right PE |
| 5 | `FLUSH1` | drive zeros and flush the array pipeline |
| 6 | `LATCH` | store the final 2x2 result matrix into `C` |
| 7+ | `IDLE` | accept the next command |

Commands issued while `busy=1` are ignored.

## Worked example

Write:

```text
A = [[ 1, 0],
     [ 0, 1]]

B = [[ 3, -2],
     [ 7,  4]]
```

Then issue the `OP_MATMUL` execute command. After `busy` returns to `0`, read back the row-major results from bank `C`:

```text
C = A x B = [[ 3, -2],
             [ 7,  4]]
```

The returned values are:

- `addr=0`: `3`
- `addr=1`: `-2`
- `addr=2`: `7`
- `addr=3`: `4`

## External hardware

No external hardware is required beyond a normal Tiny Tapeout board or simulator host.
