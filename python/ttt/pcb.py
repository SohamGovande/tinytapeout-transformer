from __future__ import annotations

import ast
import os
import time
from typing import Protocol

import serial
from serial.tools import list_ports

DEFAULT_PCB_PROJECT = "tt_um_sohamgovande_transformer"
DEFAULT_PCB_BAUDRATE = 115200
DEFAULT_PCB_TIMEOUT_S = 1.0
DEFAULT_PCB_UIO_OE_PICO = 0x5F

_BANK_IDS = {
    "A": 0,
    "B": 1,
    "C": 2,
}
_RPC_TAG = "__TTT_RPC__"
_HELPER_VERSION = 1


class RawReplSession(Protocol):
    def exec(self, script: str) -> tuple[str, str]: ...

    def close(self) -> None: ...


def _candidate_port_sort_key(port_info: list_ports.ListPortInfo) -> tuple[int, str]:
    label_parts = [
        port_info.device or "",
        port_info.description or "",
        port_info.manufacturer or "",
        port_info.product or "",
    ]
    label = " ".join(label_parts).lower()
    score = 100
    if "tiny tapeout" in label:
        score = 0
    elif "rp2350" in label:
        score = 1
    elif "rp2040" in label:
        score = 2
    elif "pico" in label:
        score = 3
    elif "usbmodem" in label or "ttyacm" in label or "ttyusb" in label:
        score = 4
    return score, port_info.device


def autodetect_pcb_port() -> str:
    ports = sorted(list_ports.comports(), key=_candidate_port_sort_key)
    candidates = []
    for port_info in ports:
        device = port_info.device or ""
        label = " ".join(
            [
                device,
                port_info.description or "",
                port_info.manufacturer or "",
                port_info.product or "",
            ]
        ).lower()
        if any(token in label for token in ("tiny tapeout", "rp2350", "rp2040", "pico", "usbmodem", "ttyacm", "ttyusb")):
            candidates.append(port_info)

    if len(candidates) == 1:
        return candidates[0].device
    if len(candidates) > 1:
        devices = ", ".join(port.device for port in candidates)
        raise RuntimeError(
            "multiple candidate Tiny Tapeout serial ports found; "
            f"set TTT_PCB_PORT or pass --pcb-port explicitly ({devices})"
        )

    if len(ports) == 1:
        return ports[0].device

    if ports:
        devices = ", ".join(port.device for port in ports)
        raise RuntimeError(
            "could not uniquely identify the Tiny Tapeout serial port; "
            f"available serial devices: {devices}"
        )
    raise RuntimeError("no serial devices were found while searching for the Tiny Tapeout board")


class MicroPythonRawRepl:
    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_PCB_BAUDRATE,
        timeout_s: float = DEFAULT_PCB_TIMEOUT_S,
    ) -> None:
        self.port = port
        self.timeout_s = float(timeout_s)
        serial_kwargs = {
            "port": port,
            "baudrate": baudrate,
            "timeout": 0.1,
            "write_timeout": self.timeout_s,
        }
        try:
            self.serial = serial.Serial(exclusive=True, **serial_kwargs)
        except TypeError:
            self.serial = serial.Serial(**serial_kwargs)
        try:
            self.serial.dtr = False
            self.serial.rts = False
        except (AttributeError, OSError):
            pass
        time.sleep(0.1)
        self._enter_raw_repl()

    def close(self) -> None:
        try:
            self.serial.write(b"\x02")
            self.serial.flush()
        except serial.SerialException:
            pass
        finally:
            self.serial.close()

    def exec(self, script: str) -> tuple[str, str]:
        payload = script.encode("utf-8")
        self._write_script(payload)
        self.serial.write(b"\x04")
        self.serial.flush()

        ack = self._read_exact(2, "raw REPL exec acknowledgement")
        if ack != b"OK":
            trailer = self._read_some()
            raise RuntimeError(
                f"unexpected raw REPL acknowledgement {ack!r} while executing script; trailing data={trailer!r}"
            )

        stdout = self._read_until(b"\x04", "raw REPL stdout")[:-1]
        stderr = self._read_until(b"\x04", "raw REPL stderr")[:-1]
        self._read_until(b">", "raw REPL prompt")
        return stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")

    def _write_script(self, payload: bytes) -> None:
        for start in range(0, len(payload), 96):
            chunk = payload[start : start + 96]
            self.serial.write(chunk)
            self.serial.flush()
            time.sleep(0.01)

    def _enter_raw_repl(self) -> None:
        self.serial.reset_input_buffer()
        self.serial.write(b"\r\x03\x03")
        self.serial.flush()
        time.sleep(0.1)
        self.serial.reset_input_buffer()
        self.serial.write(b"\r\x01")
        self.serial.flush()
        banner = self._read_until(b">", "raw REPL banner")
        if b"raw REPL" not in banner:
            raise RuntimeError(f"did not enter raw REPL successfully: {banner!r}")

    def _read_exact(self, size: int, label: str) -> bytes:
        deadline = time.monotonic() + self.timeout_s
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self.serial.read(size - len(chunks))
            if chunk:
                chunks.extend(chunk)
                continue
            if time.monotonic() > deadline:
                raise RuntimeError(f"timed out while waiting for {label}")
        return bytes(chunks)

    def _read_until(self, marker: bytes, label: str) -> bytes:
        deadline = time.monotonic() + self.timeout_s
        chunks = bytearray()
        while True:
            chunk = self.serial.read(1)
            if chunk:
                chunks.extend(chunk)
                if chunks.endswith(marker):
                    return bytes(chunks)
                continue
            if time.monotonic() > deadline:
                raise RuntimeError(f"timed out while waiting for {label}")

    def _read_some(self) -> bytes:
        data = bytearray()
        while True:
            chunk = self.serial.read(256)
            if not chunk:
                return bytes(data)
            data.extend(chunk)


def _helper_snippets(project: str, uio_oe_pico: int) -> list[str]:
    return [
        """
try:
    tt
except NameError:
    from ttboard.boot.demoboard_detect import DemoboardDetect
    from ttboard.demoboard import DemoBoard
    try:
        DemoboardDetect.probe()
    except Exception:
        pass
    tt = DemoBoard.get()
from ttboard.mode import RPMode
__ttt_project_name = PROJECT_NAME
tt.clock_project_stop()
if tt.shuttle.has(__ttt_project_name):
    getattr(tt.shuttle, __ttt_project_name).enable()
else:
    __ttt_matches = tt.shuttle.find(__ttt_project_name)
    if len(__ttt_matches) != 1:
        raise RuntimeError("project '%s' not found on this shuttle" % (__ttt_project_name,))
    __ttt_matches[0].enable()
tt.mode = RPMode.ASIC_RP_CONTROL
tt.clock_project_stop()
tt.uio_oe_pico.value = UIO_OE
tt.ui_in.value = 0
tt.uio_in.value = 0
__ttt_helper_version = HELPER_VERSION
print(RPC_TAG + "READY")
"""
        .replace("PROJECT_NAME", repr(project))
        .replace("UIO_OE", str(int(uio_oe_pico)))
        .replace("HELPER_VERSION", str(_HELPER_VERSION))
        .replace("RPC_TAG", repr(_RPC_TAG)),
        """
CMD_WRITE_A = 0
CMD_WRITE_B = 1
CMD_EXEC = 2
CMD_READ = 3
BANK_A = 0
BANK_B = 1
BANK_C = 2
OP_MATMUL = 0
OP_MATMUL_ACC = 1
OP_EW_ADD = 2
OP_EW_RELU = 3
OP_EW_SHIFT = 4
IO_W = 9
DATA_W = 5
ACC_W = 11

def __ttt_to_unsigned(value, width):
    return int(value) & ((1 << width) - 1)

def __ttt_to_signed(value, width):
    value &= (1 << width) - 1
    if value & (1 << (width - 1)):
        value -= 1 << width
    return value

def __ttt_bank_width(bank):
    if bank == BANK_C:
        return ACC_W
    return DATA_W

def __ttt_read_uio_out():
    return int(tt.uio_out.value)

def __ttt_pulse(cmd, addr, aux, payload):
    tt.ui_in.value = int(payload) & 255
    tt.uio_in.value = 1 | ((int(cmd) & 3) << 1) | ((int(addr) & 3) << 3) | ((int(aux) & 1) << 6)
    tt.clock_project_once()
    tt.ui_in.value = 0
    tt.uio_in.value = 0

def __ttt_wait_busy(max_cycles=32):
    cycles = 0
    while ((__ttt_read_uio_out() >> 5) & 1):
        tt.clock_project_once()
        cycles += 1
        if cycles > int(max_cycles):
            raise RuntimeError("timed out waiting for busy to deassert")

def __ttt_reset():
    tt.clock_project_stop()
    tt.ui_in.value = 0
    tt.uio_in.value = 0
    tt.reset_project(True)
    tt.clock_project_once()
    tt.clock_project_once()
    tt.reset_project(False)
    tt.clock_project_once()
    return 0

def __ttt_write(bank, addr, value):
    if bank == BANK_C:
        raise RuntimeError("bank C is not writable")
    width = __ttt_bank_width(bank)
    encoded = __ttt_to_unsigned(value, width)
    cmd = CMD_WRITE_B if bank == BANK_B else CMD_WRITE_A
    __ttt_pulse(cmd, addr, (encoded >> 8) & 1, encoded)
    return 0

def __ttt_read(bank, addr):
    width = __ttt_bank_width(bank)
    chunks = (width + IO_W - 1) // IO_W
    raw = 0
    for chunk in range(chunks):
        __ttt_pulse(CMD_READ, addr, 0, int(bank) | (chunk << 2))
        chunk_raw = (((__ttt_read_uio_out() >> 7) & 1) << 8) | int(tt.uo_out.value)
        used_bits = width - (chunk * IO_W)
        if used_bits > IO_W:
            used_bits = IO_W
        raw |= (chunk_raw & ((1 << used_bits) - 1)) << (chunk * IO_W)
    return __ttt_to_signed(raw, width)
print(RPC_TAG + "CORE")
"""
        .replace("RPC_TAG", repr(_RPC_TAG)),
        """
def __ttt_exec_matmul(accumulate):
    opcode = OP_MATMUL_ACC if accumulate else OP_MATMUL
    __ttt_pulse(CMD_EXEC, 0, 0, opcode)
    __ttt_wait_busy()
    return 0

def __ttt_exec_add(addr):
    __ttt_pulse(CMD_EXEC, addr, 0, OP_EW_ADD)
    __ttt_wait_busy()
    return 0

def __ttt_exec_relu(addr):
    __ttt_pulse(CMD_EXEC, addr, 0, OP_EW_RELU)
    __ttt_wait_busy()
    return 0

def __ttt_exec_shift(addr, amount):
    __ttt_pulse(CMD_EXEC, addr, 0, ((int(amount) & 15) << 4) | OP_EW_SHIFT)
    __ttt_wait_busy()
    return 0

def __ttt_write_tile(bank, values):
    __ttt_write(bank, 0, values[0])
    __ttt_write(bank, 1, values[1])
    __ttt_write(bank, 2, values[2])
    __ttt_write(bank, 3, values[3])

def __ttt_read_tile(bank):
    return [
        __ttt_read(bank, 0),
        __ttt_read(bank, 1),
        __ttt_read(bank, 2),
        __ttt_read(bank, 3),
    ]

def __ttt_shift_all(total_shift):
    remaining = int(total_shift)
    while remaining > 0:
        chunk = 15 if remaining > 15 else remaining
        __ttt_exec_shift(0, chunk)
        __ttt_exec_shift(1, chunk)
        __ttt_exec_shift(2, chunk)
        __ttt_exec_shift(3, chunk)
        remaining -= chunk

def __ttt_matmul_seq(lhs_tiles, rhs_tiles, post_shift, relu):
    if len(lhs_tiles) != len(rhs_tiles):
        raise RuntimeError("lhs/rhs tile sequence lengths must match")
    if not lhs_tiles:
        raise RuntimeError("matmul sequence must include at least one tile pair")
    for idx in range(len(lhs_tiles)):
        __ttt_write_tile(BANK_A, lhs_tiles[idx])
        __ttt_write_tile(BANK_B, rhs_tiles[idx])
        __ttt_exec_matmul(idx > 0)
    if relu:
        __ttt_exec_relu(0)
        __ttt_exec_relu(1)
        __ttt_exec_relu(2)
        __ttt_exec_relu(3)
    if int(post_shift) > 0:
        __ttt_shift_all(post_shift)
    return __ttt_read_tile(BANK_C)

def __ttt_add_tile(lhs_tile, rhs_tile, post_shift, relu):
    __ttt_write_tile(BANK_A, lhs_tile)
    __ttt_write_tile(BANK_B, rhs_tile)
    __ttt_exec_add(0)
    __ttt_exec_add(1)
    __ttt_exec_add(2)
    __ttt_exec_add(3)
    if relu:
        __ttt_exec_relu(0)
        __ttt_exec_relu(1)
        __ttt_exec_relu(2)
        __ttt_exec_relu(3)
    if int(post_shift) > 0:
        __ttt_shift_all(post_shift)
    return __ttt_read_tile(BANK_C)

print(RPC_TAG + "HELPER")
"""
        .replace("RPC_TAG", repr(_RPC_TAG)),
    ]


class PcbSa2x2Device:
    def __init__(
        self,
        port: str | None = None,
        project: str = DEFAULT_PCB_PROJECT,
        baudrate: int = DEFAULT_PCB_BAUDRATE,
        timeout_s: float = DEFAULT_PCB_TIMEOUT_S,
        uio_oe_pico: int = DEFAULT_PCB_UIO_OE_PICO,
        repl: RawReplSession | None = None,
    ) -> None:
        if repl is None:
            resolved_port = port or os.getenv("TTT_PCB_PORT") or autodetect_pcb_port()
        else:
            resolved_port = port or os.getenv("TTT_PCB_PORT") or "<injected>"
        resolved_project = os.getenv("TTT_PCB_PROJECT", project)
        resolved_baudrate = int(os.getenv("TTT_PCB_BAUDRATE", str(baudrate)))
        resolved_timeout_s = float(os.getenv("TTT_PCB_TIMEOUT", str(timeout_s)))
        self.port = resolved_port
        self.project = resolved_project
        self.uio_oe_pico = int(uio_oe_pico)
        self._owns_repl = repl is None
        self.repl = repl or MicroPythonRawRepl(
            port=resolved_port,
            baudrate=resolved_baudrate,
            timeout_s=resolved_timeout_s,
        )
        self._install_helper()
        self.reset()

    def close(self) -> None:
        try:
            try:
                self._call_ok("tt.clock_project_stop()")
            except RuntimeError:
                pass
        finally:
            if self._owns_repl:
                self.repl.close()

    def reset(self) -> None:
        self._call_ok("__ttt_reset()")

    def write_bank_entry(self, bank: str, addr: int, value: int) -> None:
        bank_id = self._bank_id(bank)
        self._call_ok(f"__ttt_write({bank_id}, {int(addr)}, {int(value)})")

    def read_bank_entry(self, bank: str, addr: int) -> int:
        bank_id = self._bank_id(bank)
        value = self._call_value(f"__ttt_read({bank_id}, {int(addr)})")
        return int(value)

    def exec_matmul(self, accumulate: bool = False) -> None:
        self._call_ok(f"__ttt_exec_matmul({1 if accumulate else 0})")

    def exec_ew_add(self, addr: int) -> None:
        self._call_ok(f"__ttt_exec_add({int(addr)})")

    def exec_ew_relu(self, addr: int) -> None:
        self._call_ok(f"__ttt_exec_relu({int(addr)})")

    def exec_ew_shift(self, addr: int, amount: int) -> None:
        self._call_ok(f"__ttt_exec_shift({int(addr)}, {int(amount)})")

    def matmul_accumulated_tiles(
        self,
        lhs_tiles: list[list[int]],
        rhs_tiles: list[list[int]],
        post_shift: int = 0,
        relu: bool = False,
    ) -> list[int]:
        lhs_text = repr([[int(value) for value in tile] for tile in lhs_tiles])
        rhs_text = repr([[int(value) for value in tile] for tile in rhs_tiles])
        result = self._call_value(
            f"__ttt_matmul_seq({lhs_text}, {rhs_text}, {int(post_shift)}, {1 if relu else 0})",
            parser=ast.literal_eval,
        )
        return [int(value) for value in result]

    def add_tile(
        self,
        lhs_tile: list[int],
        rhs_tile: list[int],
        post_shift: int = 0,
        relu: bool = False,
    ) -> list[int]:
        lhs_text = repr([int(value) for value in lhs_tile])
        rhs_text = repr([int(value) for value in rhs_tile])
        result = self._call_value(
            f"__ttt_add_tile({lhs_text}, {rhs_text}, {int(post_shift)}, {1 if relu else 0})",
            parser=ast.literal_eval,
        )
        return [int(value) for value in result]

    def _install_helper(self) -> None:
        for snippet in _helper_snippets(project=self.project, uio_oe_pico=self.uio_oe_pico):
            self._run_script(snippet)

    def _call_ok(self, expr: str) -> None:
        script = f"_ttt_result = {expr}\nprint({_RPC_TAG!r} + 'OK')"
        payload = self._run_script(script)
        if payload != "OK":
            raise RuntimeError(f"unexpected Tiny Tapeout PCB response: {payload!r}")

    def _call_value(self, expr: str, parser=None):
        script = f"_ttt_result = {expr}\nprint({_RPC_TAG!r} + repr(_ttt_result))"
        payload = self._run_script(script)
        if parser is None:
            return ast.literal_eval(payload)
        return parser(payload)

    def _run_script(self, script: str) -> str:
        stdout, stderr = self.repl.exec(script)
        stderr_clean = stderr.strip()
        if stderr_clean:
            raise RuntimeError(f"Tiny Tapeout board reported an error:\n{stderr_clean}")

        lines = [line.strip() for line in stdout.replace("\r", "").split("\n") if line.strip()]
        for line in reversed(lines):
            if line.startswith(_RPC_TAG):
                return line[len(_RPC_TAG) :]
        raise RuntimeError(f"missing Tiny Tapeout RPC marker in board response:\n{stdout.strip()}")

    def _bank_id(self, bank: str) -> int:
        try:
            return _BANK_IDS[bank]
        except KeyError as exc:
            raise ValueError(f"unknown bank {bank!r}") from exc
