"""Tiện ích dùng chung: load config, set seed, đọc/ghi TSV."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str, script: str | None = None, overrides: dict | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if script:
        cfg["script"] = script
        cfg["raw_dir"] = f"data/raw/{script}"
        cfg["processed_dir"] = f"data/processed/{script}"
        cfg["checkpoint_dir"] = f"checkpoints/{script}"
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_tsv(path: str) -> list[tuple[str, str]]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            rows.append((parts[0].strip(), parts[1].strip()))
    return rows


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
