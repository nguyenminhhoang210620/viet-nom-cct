"""CCT-C: phân loại cục bộ tại từng vị trí, giải mã song song (Mục 3.5).

score_i(v) = MLP([h_i ; E_out[v]])  (Eq 3.7-3.9), chỉ tính trên v thuộc C(xi).
Toàn bộ chuỗi được dự đoán trong một lượt truyền xuôi (Mục 3.5.2) — không tự hồi quy.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import BertEncoder
from .vocab import PAD_ID


class CCTClassifierHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(2 * d_model, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
        """h: (B,N,d) | cand_emb: (B,N,C,d) -> scores (B,N,C)."""
        h_exp = h.unsqueeze(2).expand(-1, -1, cand_emb.size(2), -1)
        u = torch.cat([h_exp, cand_emb], dim=-1)  # Eq 3.7
        g = self.dropout(self.act(self.w1(u)))  # Eq 3.8
        return self.w2(g).squeeze(-1)  # Eq 3.9


class CCTCModel(nn.Module):
    def __init__(self, vocab_size: int, enc_cfg: dict, head_cfg: dict):
        super().__init__()
        self.encoder = BertEncoder(
            vocab_size=vocab_size,
            d_model=enc_cfg["d_model"],
            n_layers=enc_cfg["n_layers"],
            n_heads=enc_cfg["n_heads"],
            d_ff=enc_cfg["d_ff"],
            dropout=enc_cfg["dropout"],
            max_len=enc_cfg["max_len"],
        )
        # Bảng nhúng đầu ra (Eq 3.6): một bảng chung trên từ vựng thống nhất, giới hạn
        # theo tập ứng viên khi tính điểm — xem "Giới hạn" trong README.
        self.output_emb = nn.Embedding(vocab_size, enc_cfg["d_model"], padding_idx=PAD_ID)
        self.head = CCTClassifierHead(enc_cfg["d_model"], head_cfg["hidden_dim"], head_cfg["dropout"])

    def forward(self, batch: dict) -> torch.Tensor:
        h = self.encoder(batch["src_ids"], batch["content_mask"])  # (B,N,d)
        cand_emb = self.output_emb(batch["cand_ids"])  # (B,N,C,d)
        scores = self.head(h, cand_emb)  # (B,N,C)
        scores = scores.masked_fill(~batch["cand_mask"], float("-inf"))
        return scores

    @torch.no_grad()
    def predict(self, batch: dict) -> torch.Tensor:
        """Giải mã song song: chọn ứng viên điểm cao nhất tại mọi vị trí (Eq 3.10).

        Trả về (B,N) id token đích dự đoán (đã ánh xạ lại từ chỉ số cục bộ trong C(xi)).
        """
        scores = self.forward(batch)
        local_idx = scores.argmax(dim=-1)  # (B,N)
        pred_ids = torch.gather(batch["cand_ids"], 2, local_idx.unsqueeze(-1)).squeeze(-1)
        return pred_ids
