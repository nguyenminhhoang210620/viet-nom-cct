"""CCT-D: bộ giải mã Transformer tự hồi quy có ràng buộc tập ứng viên (Mục 3.6).

- Đầu vào giải mã dùng ép giáo viên (teacher forcing, Eq 3.11-3.12): a_t = eBOS nếu
  t=1, ngược lại là embedding đầu ra của nhãn đúng y*_{t-1}; nhân căn(d_model), cộng
  mã hoá vị trí hình sin. Chỉ một chiều chuyển tự nên không còn embedding chiều.
- Bộ giải mã Pre-LN (Mục 3.6.1), tự chú ý có mặt nạ + chú ý chéo với H từ bộ mã hoá.
- Chấm điểm bằng tích vô hướng với bảng nhúng đầu ra (Eq 3.14), giới hạn theo C(xi).
- Không cần token EOS: số bước giải mã luôn bằng đúng độ dài chuỗi nguồn (Mục 3.3.3).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .encoder import BertEncoder
from .positional import SinusoidalPositionalEncoding
from .vocab import PAD_ID


class CCTDModel(nn.Module):
    def __init__(self, vocab_size: int, enc_cfg: dict, dec_cfg: dict):
        super().__init__()
        d_model = enc_cfg["d_model"]
        self.d_model = d_model
        self.encoder = BertEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=enc_cfg["n_layers"],
            n_heads=enc_cfg["n_heads"],
            d_ff=enc_cfg["d_ff"],
            dropout=enc_cfg["dropout"],
            max_len=enc_cfg["max_len"],
        )
        self.output_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.bos_emb = nn.Embedding(1, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=enc_cfg["max_len"])
        self.emb_dropout = nn.Dropout(dec_cfg["dropout"])
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=dec_cfg["n_heads"],
            dim_feedforward=dec_cfg["d_ff"],
            dropout=dec_cfg["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN (Mục 3.6.1)
        )
        # Pre-LN không tự chuẩn hoá đầu ra của lớp cuối (residual stream có thể tăng dần
        # qua các lớp) -> cần một LayerNorm cuối cùng, nếu không tích vô hướng chấm điểm
        # (Eq 3.14) có thể bùng nổ về độ lớn. Đây là thực hành chuẩn cho Pre-LN Transformer.
        self.decoder = nn.TransformerDecoder(layer, num_layers=dec_cfg["n_layers"], norm=nn.LayerNorm(d_model))

    # ---- Huấn luyện: ép giáo viên trên cả chuỗi cùng lúc (vector hoá) ----
    def _teacher_forcing_input(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        B, N = tgt_ids.shape
        bos = self.bos_emb(torch.zeros(B, 1, dtype=torch.long, device=tgt_ids.device))  # (B,1,d)
        prev = self.output_emb(tgt_ids[:, :-1]) if N > 1 else tgt_ids.new_zeros(B, 0, self.d_model, dtype=bos.dtype)
        a = torch.cat([bos, prev], dim=1)  # (B,N,d), a_t Eq 3.11
        a = a * math.sqrt(self.d_model)  # Eq 3.12
        positions = torch.arange(N, device=tgt_ids.device).unsqueeze(0).expand(B, -1)
        a = a + self.pos_enc(positions)
        return self.emb_dropout(a)

    def forward(self, batch: dict) -> torch.Tensor:
        H = self.encoder(batch["src_ids"], batch["content_mask"])  # (B,N,d)
        content_mask = batch["content_mask"]
        B, N = content_mask.shape
        a = self._teacher_forcing_input(batch["tgt_ids"])
        causal_mask = nn.Transformer.generate_square_subsequent_mask(N).to(a.device)
        dec_out = self.decoder(
            tgt=a,
            memory=H,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~content_mask,
            memory_key_padding_mask=~content_mask,
        )  # (B,N,d)
        cand_emb = self.output_emb(batch["cand_ids"])  # (B,N,C,d)
        scores = torch.einsum("bnd,bncd->bnc", dec_out, cand_emb)  # Eq 3.14
        scores = scores.masked_fill(~batch["cand_mask"], float("-inf"))
        return scores

    # ---- Suy luận: tìm kiếm chùm tia, giải mã tuần tự (Mục 3.6.4, 3.8.3) ----
    def _decoder_step_input(self, seq: list[int], device) -> torch.Tensor:
        t = len(seq)
        bos = self.bos_emb(torch.zeros(1, 1, dtype=torch.long, device=device))
        if t > 0:
            prev = self.output_emb(torch.tensor([seq], device=device))
            a = torch.cat([bos, prev], dim=1)
        else:
            a = bos
        a = a * math.sqrt(self.d_model)
        positions = torch.arange(t + 1, device=device).unsqueeze(0)
        return a + self.pos_enc(positions)

    @torch.no_grad()
    def generate_beam(
        self,
        src_ids: list[int],
        cand_ids_per_pos: list[list[int]],
        beam_size: int,
        device,
    ) -> list[int]:
        """Trả về danh sách id token đích, độ dài đúng bằng len(src_ids).

        LƯU Ý: cài đặt tham khảo, không dùng KV-cache nên chạy lại toàn bộ decoder mỗi
        bước — đủ dùng để kiểm chứng logic và với câu ngắn/trung bình, nhưng chậm hơn một
        cài đặt tối ưu tốc độ cho suy luận quy mô lớn.
        """
        self.eval()
        N = len(src_ids)
        if N == 0:
            return []
        src_t = torch.tensor([src_ids], device=device)
        content_mask = torch.ones((1, N), dtype=torch.bool, device=device)
        H = self.encoder(src_t, content_mask)

        beams: list[tuple[list[int], float]] = [([], 0.0)]
        for t in range(N):
            cand_ids_t = cand_ids_per_pos[t] or [PAD_ID]
            cand_t = torch.tensor(cand_ids_t, device=device)
            expanded = []
            for seq, score in beams:
                a = self._decoder_step_input(seq, device)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(t + 1).to(device)
                dec_out = self.decoder(tgt=a, memory=H, tgt_mask=causal_mask, memory_key_padding_mask=~content_mask)
                last = dec_out[0, -1]  # (d,)
                cand_emb = self.output_emb(cand_t)  # (C,d)
                logprobs = torch.log_softmax(cand_emb @ last, dim=-1)
                k = min(beam_size, logprobs.size(0))
                vals, idxs = logprobs.topk(k)
                for v, idx in zip(vals.tolist(), idxs.tolist()):
                    expanded.append((seq + [cand_ids_t[idx]], score + v))
            expanded.sort(key=lambda x: -x[1])
            beams = expanded[:beam_size]
        best_seq, _ = beams[0]
        return best_seq
