from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PACKAGE_ROOT / "data" / "tinyshakespeare.txt"


def ensure_dataset(path: str | Path | None = None) -> Path:
    data_path = Path(path) if path is not None else DATA_PATH
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists() and data_path.stat().st_size > 0:
        return data_path
    with urllib.request.urlopen(DATA_URL) as response:
        data_path.write_bytes(response.read())
    return data_path
