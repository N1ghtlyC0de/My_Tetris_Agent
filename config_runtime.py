from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


RUNTIME_CONFIG_PATH = Path("configs/runtime_config.json")


def save_runtime_config(runtime: Dict[str, Any]) -> None:
    RUNTIME_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)


def load_runtime_config() -> Dict[str, Any]:
    if not RUNTIME_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Runtime config not found at {RUNTIME_CONFIG_PATH.as_posix()}. "
            "Run calibrate.py first."
        )
    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)