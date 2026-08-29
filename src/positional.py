"""Mã hoá vị trí hình sin cố định (Eq 2.1), dùng cho bộ giải mã CCT-D.

Bộ mã hoá BERT dùng embedding vị trí HỌC ĐƯỢC (Mục 2.3.1) — cài trực tiếp trong
encoder.py bằng nn.Embedding, không dùng module này.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: (B, T) long -> (B, T, d_model)."""
        return self.pe[positions]
