from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import torch

from .quantization import INT5_MAX, INT5_MIN, wrap_signed_scalar

BANK_A = "A"
BANK_B = "B"
BANK_C = "C"


class Sa2x2Device(Protocol):
    def reset(self) -> None: ...

    def close(self) -> None: ...

    def write_bank_entry(self, bank: str, addr: int, value: int) -> None: ...

    def read_bank_entry(self, bank: str, addr: int) -> int: ...

    def exec_matmul(self, accumulate: bool = False) -> None: ...

    def exec_ew_add(self, addr: int) -> None: ...

    def exec_ew_relu(self, addr: int) -> None: ...

    def exec_ew_shift(self, addr: int, amount: int) -> None: ...


class ReferenceSa2x2Device:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.bank_a = [0, 0, 0, 0]
        self.bank_b = [0, 0, 0, 0]
        self.bank_c = [0, 0, 0, 0]

    def close(self) -> None:
        return None

    def write_bank_entry(self, bank: str, addr: int, value: int) -> None:
        wrapped = wrap_signed_scalar(int(value), 5)
        if bank == BANK_A:
            self.bank_a[addr] = wrapped
        elif bank == BANK_B:
            self.bank_b[addr] = wrapped
        else:
            raise ValueError(f"bank {bank} is not writable")

    def read_bank_entry(self, bank: str, addr: int) -> int:
        if bank == BANK_A:
            return self.bank_a[addr]
        if bank == BANK_B:
            return self.bank_b[addr]
        if bank == BANK_C:
            return self.bank_c[addr]
        raise ValueError(f"unknown bank {bank}")

    def exec_matmul(self, accumulate: bool = False) -> None:
        a = [[self.bank_a[0], self.bank_a[1]], [self.bank_a[2], self.bank_a[3]]]
        b = [[self.bank_b[0], self.bank_b[1]], [self.bank_b[2], self.bank_b[3]]]
        result = [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ]
        if accumulate:
            self.bank_c = [
                wrap_signed_scalar(self.bank_c[idx] + result[idx], 11)
                for idx in range(4)
            ]
        else:
            self.bank_c = [wrap_signed_scalar(value, 11) for value in result]

    def exec_ew_add(self, addr: int) -> None:
        self.bank_c[addr] = wrap_signed_scalar(self.bank_a[addr] + self.bank_b[addr], 11)

    def exec_ew_relu(self, addr: int) -> None:
        self.bank_c[addr] = 0 if self.bank_c[addr] < 0 else self.bank_c[addr]

    def exec_ew_shift(self, addr: int, amount: int) -> None:
        if amount >= 11:
            self.bank_c[addr] = -1 if self.bank_c[addr] < 0 else 0
        else:
            self.bank_c[addr] = self.bank_c[addr] >> amount


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _chip_server_binary(build_dir: Path | None = None) -> Path:
    root = _repo_root()
    actual_build_dir = build_dir or (root / "python" / "build" / "chip_server")
    return actual_build_dir / "Vttt_chip_server"


def build_chip_server(build_dir: Path | None = None) -> Path:
    if shutil.which("verilator") is None:
        raise RuntimeError("verilator was not found on PATH")

    root = _repo_root()
    actual_build_dir = build_dir or (root / "python" / "build" / "chip_server")
    actual_build_dir.mkdir(parents=True, exist_ok=True)
    binary = _chip_server_binary(actual_build_dir)

    source_paths = [
        root / "src" / "tt_um_sohamgovande_transformer.sv",
        root / "src" / "control" / "sa2x2_controller.sv",
        root / "src" / "core" / "pe_mac.sv",
        root / "src" / "core" / "systolic_array_2x2.sv",
        root / "python" / "csrc" / "ttt_chip_server.cpp",
    ]
    latest_source_mtime = max(path.stat().st_mtime for path in source_paths)
    if binary.exists() and binary.stat().st_mtime >= latest_source_mtime:
        return binary

    cmd = [
        "verilator",
        "--cc",
        "--exe",
        "--build",
        "--sv",
        "--timing",
        "-Wno-TIMESCALEMOD",
        "--Mdir",
        str(actual_build_dir),
        "--top-module",
        "tt_um_sohamgovande_transformer",
        "-CFLAGS",
        f"-I{root}",
        "-o",
        binary.name,
        str(root / "src" / "tt_um_sohamgovande_transformer.sv"),
        str(root / "src" / "control" / "sa2x2_controller.sv"),
        str(root / "src" / "core" / "pe_mac.sv"),
        str(root / "src" / "core" / "systolic_array_2x2.sv"),
        str(root / "python" / "csrc" / "ttt_chip_server.cpp"),
    ]
    subprocess.run(cmd, cwd=root, check=True)
    return binary


class VerilatedSa2x2Device:
    def __init__(self, build_dir: Path | None = None):
        binary = build_chip_server(build_dir=build_dir)
        self.proc = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=_repo_root(),
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("failed to open chip server pipes")
        self.reset()

    def _exchange(self, line: str) -> str:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("chip server is not available")
        self.proc.stdin.write(f"{line}\n")
        self.proc.stdin.flush()
        response = self.proc.stdout.readline()
        if response == "":
            stderr_output = ""
            if self.proc.stderr is not None:
                stderr_output = self.proc.stderr.read().strip()
            raise RuntimeError(f"chip server exited unexpectedly: {stderr_output}")
        response = response.strip()
        if response.startswith("ERR "):
            raise RuntimeError(response[4:])
        return response

    def reset(self) -> None:
        response = self._exchange("reset")
        if response != "OK":
            raise RuntimeError(f"unexpected reset response: {response}")

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self._exchange("shutdown")
        except RuntimeError:
            pass
        finally:
            self.proc.wait(timeout=5)

    def write_bank_entry(self, bank: str, addr: int, value: int) -> None:
        response = self._exchange(f"write_{bank.lower()} {addr} {value}")
        if response != "OK":
            raise RuntimeError(f"unexpected write response: {response}")

    def read_bank_entry(self, bank: str, addr: int) -> int:
        response = self._exchange(f"read_{bank.lower()} {addr}")
        if not response.startswith("VALUE "):
            raise RuntimeError(f"unexpected read response: {response}")
        return int(response.split()[1])

    def exec_matmul(self, accumulate: bool = False) -> None:
        command = "matmul_acc" if accumulate else "matmul"
        response = self._exchange(command)
        if response != "OK":
            raise RuntimeError(f"unexpected matmul response: {response}")

    def exec_ew_add(self, addr: int) -> None:
        response = self._exchange(f"ew_add {addr}")
        if response != "OK":
            raise RuntimeError(f"unexpected add response: {response}")

    def exec_ew_relu(self, addr: int) -> None:
        response = self._exchange(f"ew_relu {addr}")
        if response != "OK":
            raise RuntimeError(f"unexpected relu response: {response}")

    def exec_ew_shift(self, addr: int, amount: int) -> None:
        response = self._exchange(f"ew_shift {addr} {amount}")
        if response != "OK":
            raise RuntimeError(f"unexpected shift response: {response}")


class Int5TiledRuntime:
    def __init__(self, device: Sa2x2Device):
        self.device = device

    def close(self) -> None:
        self.device.close()

    def _extract_tile(self, matrix: torch.Tensor, row0: int, col0: int) -> list[int]:
        tile: list[int] = []
        rows, cols = matrix.shape
        for r in range(2):
            for c in range(2):
                rr = row0 + r
                cc = col0 + c
                if rr < rows and cc < cols:
                    tile.append(int(matrix[rr, cc].item()))
                else:
                    tile.append(0)
        return tile

    def _store_tile(self, out: torch.Tensor, row0: int, col0: int, tile: list[int]) -> None:
        rows, cols = out.shape
        idx = 0
        for r in range(2):
            for c in range(2):
                rr = row0 + r
                cc = col0 + c
                if rr < rows and cc < cols:
                    out[rr, cc] = int(tile[idx])
                idx += 1

    def _write_tile(self, bank: str, tile: list[int]) -> None:
        for addr, value in enumerate(tile):
            self.device.write_bank_entry(bank, addr, value)

    def _read_tile(self, bank: str) -> list[int]:
        return [self.device.read_bank_entry(bank, addr) for addr in range(4)]

    def _apply_shift_to_c(self, total_shift: int) -> None:
        remaining = int(total_shift)
        while remaining > 0:
            chunk = min(remaining, 15)
            for addr in range(4):
                self.device.exec_ew_shift(addr, chunk)
            remaining -= chunk

    def matmul(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        post_shift: int = 0,
        relu: bool = False,
        clamp_output_to_int5: bool = True,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lhs.ndim != 2 or rhs.ndim != 2:
            raise ValueError("matmul expects rank-2 tensors")
        lhs_i = lhs.to(torch.int32)
        rhs_i = rhs.to(torch.int32)
        if lhs_i.shape[1] != rhs_i.shape[0]:
            raise ValueError("matmul inner dimensions must match")

        out = torch.zeros((lhs_i.shape[0], rhs_i.shape[1]), dtype=torch.int32)

        for row0 in range(0, lhs_i.shape[0], 2):
            for col0 in range(0, rhs_i.shape[1], 2):
                first_k_tile = True
                for k0 in range(0, lhs_i.shape[1], 2):
                    self._write_tile(BANK_A, self._extract_tile(lhs_i, row0, k0))
                    self._write_tile(BANK_B, self._extract_tile(rhs_i, k0, col0))
                    self.device.exec_matmul(accumulate=not first_k_tile)
                    first_k_tile = False

                if relu:
                    for addr in range(4):
                        self.device.exec_ew_relu(addr)
                if post_shift > 0:
                    self._apply_shift_to_c(post_shift)

                tile = self._read_tile(BANK_C)
                self._store_tile(out, row0, col0, tile)

        if mask is not None:
            out = torch.where(mask.to(torch.bool), out, torch.zeros_like(out))

        if clamp_output_to_int5:
            out = out.clamp(INT5_MIN, INT5_MAX)

        return out.to(torch.int32)

    def add(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        post_shift: int = 0,
        relu: bool = False,
        clamp_output_to_int5: bool = True,
    ) -> torch.Tensor:
        if lhs.shape != rhs.shape:
            raise ValueError("add expects matching shapes")
        lhs_i = lhs.to(torch.int32)
        rhs_i = rhs.to(torch.int32)
        out = torch.zeros_like(lhs_i)

        for row0 in range(0, lhs_i.shape[0], 2):
            for col0 in range(0, lhs_i.shape[1], 2):
                self._write_tile(BANK_A, self._extract_tile(lhs_i, row0, col0))
                self._write_tile(BANK_B, self._extract_tile(rhs_i, row0, col0))
                for addr in range(4):
                    self.device.exec_ew_add(addr)
                if relu:
                    for addr in range(4):
                        self.device.exec_ew_relu(addr)
                if post_shift > 0:
                    self._apply_shift_to_c(post_shift)
                tile = self._read_tile(BANK_C)
                self._store_tile(out, row0, col0, tile)

        if clamp_output_to_int5:
            out = out.clamp(INT5_MIN, INT5_MAX)

        return out.to(torch.int32)

    def relu(
        self,
        values: torch.Tensor,
        clamp_output_to_int5: bool = True,
    ) -> torch.Tensor:
        zeros = torch.zeros_like(values, dtype=torch.int32)
        return self.add(values, zeros, relu=True, clamp_output_to_int5=clamp_output_to_int5)
