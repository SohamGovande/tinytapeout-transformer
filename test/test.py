import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


CMD_WRITE_A = 0b00
CMD_WRITE_B = 0b01
CMD_EXEC = 0b10
CMD_READ = 0b11

BANK_A = 0
BANK_B = 1
BANK_C = 2

OP_MATMUL = 0b000
OP_MATMUL_ACC = 0b001
OP_EW_ADD = 0b010
OP_EW_RELU = 0b011
OP_EW_SHIFT = 0b100

IO_W = 9
DATA_W = int(os.environ.get("SA2X2_DATA_W", "4"))
ACC_W = int(os.environ.get("SA2X2_ACC_W", str((2 * DATA_W) + 1)))


def to_unsigned(value: int, width: int) -> int:
    return value & ((1 << width) - 1)


def to_signed(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    if value & (1 << (width - 1)):
        value -= 1 << width
    return value


def min_signed(width: int) -> int:
    return -(1 << (width - 1))


def max_signed(width: int) -> int:
    return (1 << (width - 1)) - 1


def bank_width(bank: int) -> int:
    return ACC_W if bank == BANK_C else DATA_W


def matrix_product(a, b):
    return [
        [sum(a[row][k] * b[k][col] for k in range(2)) for col in range(2)]
        for row in range(2)
    ]


def encode_exec(op: int, shift: int = 0) -> int:
    return ((shift & 0xF) << 4) | (op & 0x7)


def get_busy(dut) -> int:
    return (dut.uio_out.value.to_unsigned() >> 5) & 0x1


def get_read_chunk(dut) -> int:
    raw = ((dut.uio_out.value.to_unsigned() >> 7) & 0x1) << 8
    raw |= dut.uo_out.value.to_unsigned()
    return raw


async def setup_dut(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 2)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()


async def pulse_cmd(dut, cmd: int, addr: int = 0, aux: int = 0, payload: int = 0):
    await FallingEdge(dut.clk)
    dut.ui_in.value = to_unsigned(payload, 8)
    dut.uio_in.value = (
        0x1
        | ((cmd & 0x3) << 1)
        | ((addr & 0x3) << 3)
        | ((aux & 0x1) << 6)
    )
    await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    await ReadOnly()


async def write_bank_entry(dut, bank: int, addr: int, value: int):
    width = bank_width(bank)
    encoded = to_unsigned(value, width)
    cmd = CMD_WRITE_B if bank == BANK_B else CMD_WRITE_A
    await pulse_cmd(
        dut,
        cmd,
        addr=addr,
        aux=(encoded >> 8) & 0x1,
        payload=encoded,
    )


async def read_bank_entry(dut, bank: int, addr: int) -> int:
    width = bank_width(bank)
    raw = 0
    chunks = (width + IO_W - 1) // IO_W

    for chunk in range(chunks):
        await pulse_cmd(dut, CMD_READ, addr=addr, payload=bank | (chunk << 2))
        chunk_raw = get_read_chunk(dut)
        used_bits = min(IO_W, width - (chunk * IO_W))
        raw |= (chunk_raw & ((1 << used_bits) - 1)) << (chunk * IO_W)

    return to_signed(raw, width)


async def write_matrix(dut, bank: int, matrix):
    for addr in range(4):
        row = addr >> 1
        col = addr & 0x1
        await write_bank_entry(dut, bank, addr, matrix[row][col])


async def read_matrix(dut, bank: int):
    values = []
    for addr in range(4):
        values.append(await read_bank_entry(dut, bank, addr))
    return [[values[0], values[1]], [values[2], values[3]]]


async def exec_op(dut, op: int, addr: int = 0, shift: int = 0):
    await pulse_cmd(dut, CMD_EXEC, addr=addr, payload=encode_exec(op, shift))


async def wait_while_busy(dut, max_cycles: int = 16):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if not get_busy(dut):
            return
    raise AssertionError("timed out waiting for busy to deassert")


@cocotb.test()
async def test_reset_smoke(dut):
    await setup_dut(dut)

    assert dut.uo_out.value.to_unsigned() == 0
    assert dut.uio_out.value.to_unsigned() == 0
    assert dut.uio_oe.value.to_unsigned() == 0xA0
    assert get_busy(dut) == 0


@cocotb.test()
async def test_bank_write_and_readback(dut):
    await setup_dut(dut)

    lo = min_signed(DATA_W)
    hi = max_signed(DATA_W)
    values_a = [hi, lo, 1, -2]
    values_b = [-1, hi - 1, lo + 1, 3]

    for addr, value in enumerate(values_a):
        await write_bank_entry(dut, BANK_A, addr, value)

    for addr, value in enumerate(values_b):
        await write_bank_entry(dut, BANK_B, addr, value)

    for addr, value in enumerate(values_a):
        assert await read_bank_entry(dut, BANK_A, addr) == value

    for addr, value in enumerate(values_b):
        assert await read_bank_entry(dut, BANK_B, addr) == value

    for addr in range(4):
        assert await read_bank_entry(dut, BANK_C, addr) == 0


@cocotb.test()
async def test_matmul_populates_result_bank(dut):
    await setup_dut(dut)

    a = [[1, -2], [3, 4]]
    b = [[-1, 5], [2, -3]]
    expected = matrix_product(a, b)

    await write_matrix(dut, BANK_A, a)
    await write_matrix(dut, BANK_B, b)
    await exec_op(dut, OP_MATMUL)
    await wait_while_busy(dut)

    assert await read_matrix(dut, BANK_C) == expected
    assert await read_matrix(dut, BANK_A) == a
    assert await read_matrix(dut, BANK_B) == b


@cocotb.test()
async def test_matmul_accumulates_into_result_bank(dut):
    await setup_dut(dut)

    a = [[1, 2], [3, -1]]
    b = [[4, 0], [-2, 5]]
    expected = matrix_product(a, b)

    await write_matrix(dut, BANK_A, a)
    await write_matrix(dut, BANK_B, b)

    await exec_op(dut, OP_MATMUL)
    await wait_while_busy(dut)
    assert await read_matrix(dut, BANK_C) == expected

    await exec_op(dut, OP_MATMUL_ACC)
    await wait_while_busy(dut)

    doubled = [[2 * value for value in row] for row in expected]
    assert await read_matrix(dut, BANK_C) == doubled


@cocotb.test()
async def test_elementwise_add_writes_selected_result_word(dut):
    await setup_dut(dut)

    a = [[2, -3], [max_signed(DATA_W) // 2, 1]]
    b = [[-1, 4], [-(max_signed(DATA_W) // 3), -2]]
    expected = [[a[row][col] + b[row][col] for col in range(2)] for row in range(2)]

    await write_matrix(dut, BANK_A, a)
    await write_matrix(dut, BANK_B, b)

    for addr in range(4):
        await exec_op(dut, OP_EW_ADD, addr=addr)
        await wait_while_busy(dut)

    assert await read_matrix(dut, BANK_C) == expected


@cocotb.test()
async def test_relu_and_shift_are_in_place_on_result_bank(dut):
    await setup_dut(dut)

    positive_hi = max_signed(DATA_W)
    positive_lo = min(positive_hi, 7)
    a = [[-3, positive_hi], [-1, positive_lo]]
    zeros = [[0, 0], [0, 0]]

    await write_matrix(dut, BANK_A, a)
    await write_matrix(dut, BANK_B, zeros)

    for addr in range(4):
        await exec_op(dut, OP_EW_ADD, addr=addr)
        await wait_while_busy(dut)

    for addr in range(4):
        await exec_op(dut, OP_EW_RELU, addr=addr)
        await wait_while_busy(dut)

    relu_expected = [[0, positive_hi], [0, positive_lo]]
    assert await read_matrix(dut, BANK_C) == relu_expected

    for addr in range(4):
        await exec_op(dut, OP_EW_SHIFT, addr=addr, shift=1)
        await wait_while_busy(dut)

    shifted_expected = [[0, positive_hi >> 1], [0, positive_lo >> 1]]
    assert await read_matrix(dut, BANK_C) == shifted_expected


@cocotb.test()
async def test_busy_ignores_overlapping_commands(dut):
    await setup_dut(dut)

    a = [[1, 0], [0, 1]]
    b = [[6, -2], [5, 4]]
    overwrite_value = min_signed(DATA_W) + 1

    await write_matrix(dut, BANK_A, a)
    await write_matrix(dut, BANK_B, b)
    await read_bank_entry(dut, BANK_C, 0)

    await exec_op(dut, OP_MATMUL)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert get_busy(dut) == 1

    await write_bank_entry(dut, BANK_A, 0, overwrite_value)
    await pulse_cmd(dut, CMD_READ, addr=3, payload=BANK_B)
    await exec_op(dut, OP_EW_ADD, addr=2)

    await wait_while_busy(dut)

    assert await read_bank_entry(dut, BANK_A, 0) == 1
    assert await read_bank_entry(dut, BANK_C, 3) == 4
