"""Bộ mã hoá BERT hai chiều, huấn luyện từ đầu (Mục 2.3, 3.4.2).

Không dùng trọng số tiền huấn luyện (không có checkpoint phù hợp cho từ vựng hỗn
hợp Hán/Nôm + Quốc ngữ, Mục 2.3.2). Kiến trúc Post-LN chuẩn BERT (Eq 2.2), embedding
vị trí HỌC ĐƯỢC (Mục 2.3.1) — không dùng mã hoá vị trí hình sin ở đây.

Mô hình chỉ có MỘT chiều chuyển tự (Quốc ngữ -> Hán/Nôm), nên không còn token chỉ thị
chiều ở đầu chuỗi: H = (h1, ..., hn) tương ứng trực tiếp các token nguồn x1..xn.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .vocab import PAD_ID


class BertEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float, max_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_ln = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,  # Post-LN (Eq 2.2), giống BERT gốc
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, src_ids: torch.Tensor, content_mask: torch.Tensor) -> torch.Tensor:
        """src_ids: (B,N) | content_mask: (B,N) bool, True=vị trí thật.

        Trả về H: (B, N, d_model), tương ứng trực tiếp các vị trí nguồn x1..xn.
        """
        B, N = src_ids.shape
        if N > self.max_len:
            raise ValueError(f"Chuỗi dài {N} vượt quá max_len={self.max_len}; tăng encoder.max_len trong config.")
        positions = torch.arange(N, device=src_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_emb(src_ids) + self.pos_emb(positions)
        x = self.emb_ln(x)
        x = self.emb_dropout(x)

        key_padding_mask = ~content_mask  # nn.TransformerEncoder: True = bỏ qua (đệm)
        H = self.layers(x, src_key_padding_mask=key_padding_mask)
        return H
