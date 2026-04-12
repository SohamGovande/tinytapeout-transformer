from __future__ import annotations

import torch

from ttt.pcb import PcbSa2x2Device
from ttt.runtime import BANK_A, BANK_B, BANK_C, Int5TiledRuntime, ReferenceSa2x2Device


class FastOnlyReferenceDevice:
    def close(self) -> None:
        return None

    def reset(self) -> None:
        return None

    def write_bank_entry(self, bank: str, addr: int, value: int) -> None:
        raise AssertionError("runtime should have used the fast tile matmul/add path")

    def read_bank_entry(self, bank: str, addr: int) -> int:
        raise AssertionError("runtime should have used the fast tile matmul/add path")

    def exec_matmul(self, accumulate: bool = False) -> None:
        raise AssertionError("runtime should have used the fast tile matmul/add path")

    def exec_ew_add(self, addr: int) -> None:
        raise AssertionError("runtime should have used the fast tile matmul/add path")

    def exec_ew_relu(self, addr: int) -> None:
        raise AssertionError("runtime should have used the fast tile matmul/add path")

    def exec_ew_shift(self, addr: int, amount: int) -> None:
        raise AssertionError("runtime should have used the fast tile matmul/add path")

    def matmul_accumulated_tiles(
        self,
        lhs_tiles: list[list[int]],
        rhs_tiles: list[list[int]],
        post_shift: int = 0,
        relu: bool = False,
    ) -> list[int]:
        ref = ReferenceSa2x2Device()
        for idx, (lhs_tile, rhs_tile) in enumerate(zip(lhs_tiles, rhs_tiles)):
            for addr, value in enumerate(lhs_tile):
                ref.write_bank_entry(BANK_A, addr, value)
            for addr, value in enumerate(rhs_tile):
                ref.write_bank_entry(BANK_B, addr, value)
            ref.exec_matmul(accumulate=idx > 0)
        if relu:
            for addr in range(4):
                ref.exec_ew_relu(addr)
        remaining = int(post_shift)
        while remaining > 0:
            chunk = min(remaining, 15)
            for addr in range(4):
                ref.exec_ew_shift(addr, chunk)
            remaining -= chunk
        return [ref.read_bank_entry(BANK_C, addr) for addr in range(4)]

    def add_tile(
        self,
        lhs_tile: list[int],
        rhs_tile: list[int],
        post_shift: int = 0,
        relu: bool = False,
    ) -> list[int]:
        ref = ReferenceSa2x2Device()
        for addr, value in enumerate(lhs_tile):
            ref.write_bank_entry(BANK_A, addr, value)
        for addr, value in enumerate(rhs_tile):
            ref.write_bank_entry(BANK_B, addr, value)
        for addr in range(4):
            ref.exec_ew_add(addr)
        if relu:
            for addr in range(4):
                ref.exec_ew_relu(addr)
        remaining = int(post_shift)
        while remaining > 0:
            chunk = min(remaining, 15)
            for addr in range(4):
                ref.exec_ew_shift(addr, chunk)
            remaining -= chunk
        return [ref.read_bank_entry(BANK_C, addr) for addr in range(4)]


class FakeRawRepl:
    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.closed = False

    def exec(self, script: str) -> tuple[str, str]:
        self.scripts.append(script)
        if "__TTT_RPC__READY" in script:
            return "ttboard.project_mux: Enable design tt_um_sohamgovande_transformer\n__TTT_RPC__READY\n", ""
        if "__TTT_RPC__CORE" in script:
            return "__TTT_RPC__CORE\n", ""
        if "__TTT_RPC__HELPER" in script:
            return "__TTT_RPC__HELPER\n", ""
        if "__ttt_read(2, 3)" in script:
            return "__TTT_RPC__-7\n", ""
        if "__ttt_matmul_seq" in script:
            return "__TTT_RPC__[1, -2, 3, 4]\n", ""
        if "__ttt_add_tile" in script:
            return "__TTT_RPC__[0, 1, 2, 3]\n", ""
        if "__ttt_exec_shift(1, 4)" in script:
            return "log noise\n__TTT_RPC__OK\n", ""
        return "__TTT_RPC__OK\n", ""

    def close(self) -> None:
        self.closed = True


def test_runtime_uses_fast_tile_paths_when_device_supports_them() -> None:
    lhs = torch.tensor(
        [
            [1, -2, 3, 0],
            [2, 1, -1, 4],
            [0, 2, 3, -2],
        ],
        dtype=torch.int32,
    )
    rhs = torch.tensor(
        [
            [2, 1, -1, 0],
            [3, -2, 1, 4],
            [1, 0, 2, -3],
            [2, 1, -2, 1],
        ],
        dtype=torch.int32,
    )
    add_lhs = torch.tensor([[3, -4, 2], [1, 5, -6]], dtype=torch.int32)
    add_rhs = torch.tensor([[1, 2, -3], [-2, 1, 4]], dtype=torch.int32)

    expected_runtime = Int5TiledRuntime(ReferenceSa2x2Device())
    fast_runtime = Int5TiledRuntime(FastOnlyReferenceDevice())
    try:
        expected_mm = expected_runtime.matmul(lhs, rhs, post_shift=2, relu=True, clamp_output_to_int5=True)
        got_mm = fast_runtime.matmul(lhs, rhs, post_shift=2, relu=True, clamp_output_to_int5=True)

        expected_add = expected_runtime.add(add_lhs, add_rhs, post_shift=1, relu=True, clamp_output_to_int5=True)
        got_add = fast_runtime.add(add_lhs, add_rhs, post_shift=1, relu=True, clamp_output_to_int5=True)
    finally:
        expected_runtime.close()
        fast_runtime.close()

    assert torch.equal(got_mm, expected_mm)
    assert torch.equal(got_add, expected_add)


def test_pcb_device_installs_helper_and_formats_rpc_calls() -> None:
    fake_repl = FakeRawRepl()
    device = PcbSa2x2Device(port="/dev/fake", repl=fake_repl)
    try:
        assert len(fake_repl.scripts) >= 4
        assert "tt_um_sohamgovande_transformer" in fake_repl.scripts[0]
        assert "tt.uio_oe_pico.value = 95" in fake_repl.scripts[0]

        device.write_bank_entry(BANK_A, 2, -3)
        assert "__ttt_write(0, 2, -3)" in fake_repl.scripts[-1]

        value = device.read_bank_entry(BANK_C, 3)
        assert value == -7
        assert "__ttt_read(2, 3)" in fake_repl.scripts[-1]

        tile = device.matmul_accumulated_tiles(
            lhs_tiles=[[1, 2, 3, 4], [5, 6, 7, 8]],
            rhs_tiles=[[8, 7, 6, 5], [4, 3, 2, 1]],
            post_shift=2,
            relu=True,
        )
        assert tile == [1, -2, 3, 4]
        assert "__ttt_matmul_seq" in fake_repl.scripts[-1]

        add_tile = device.add_tile([1, 0, -1, 2], [2, 3, 4, 5], post_shift=1, relu=False)
        assert add_tile == [0, 1, 2, 3]
        assert "__ttt_add_tile" in fake_repl.scripts[-1]

        device.exec_ew_shift(1, 4)
        assert "__ttt_exec_shift(1, 4)" in fake_repl.scripts[-1]
    finally:
        device.close()

    assert fake_repl.closed is False
