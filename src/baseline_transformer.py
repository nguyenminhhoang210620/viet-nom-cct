"""Transformer cơ sở (baseline) KHÔNG ràng buộc tập ứng viên (Mục 4.3.1).

Dùng để đối chứng: cùng dữ liệu, cùng cách chia tập, cùng quy mô kiến trúc với
CCT-C/CCT-D, nhưng sinh chuỗi tự do trên toàn bộ từ vựng đích. Báo cáo dùng fairseq;
bản này viết lại bằng PyTorch thuần để có con số đối chứng tương đối — không kỳ vọng
khớp tuyệt đối số liệu trong báo cáo (xem README, mục Giới hạn).

Khác với CCT-C/CCT-D, baseline có EOS nên độ dài đầu ra có thể khác độ dài nguồn.
"""
from __future__ import annotations

import json
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .positional import SinusoidalPositionalEncoding
from .vocab import PAD_ID, Vocab

BOS, EOS = "[BOS]", "[EOS]"


class BaselineDataset(Dataset):
    """Một chiều duy nhất (Quốc ngữ -> Hán/Nôm); thêm BOS/EOS ở phía đích."""

    def __init__(self, jsonl_path: str, vocab: Vocab, bos_id: int, eos_id: int, max_len: int = 128):
        self.examples = []
        self.vocab = vocab
        self.bos_id = bos_id
        self.eos_id = eos_id
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                qn, hn = row["qn"], row["hn"]
                if not qn or not hn or len(qn) > max_len - 2 or len(hn) > max_len - 2:
                    continue
                self.examples.append((qn, hn))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src_toks, tgt_toks = self.examples[idx]
        src_ids = [self.vocab.encode(t) for t in src_toks]
        tgt_ids = [self.bos_id] + [self.vocab.encode(t) for t in tgt_toks] + [self.eos_id]
        return src_ids, tgt_ids


def collate_baseline(batch, pad_id: int = PAD_ID):
    Ns = max(len(s) for s, _ in batch)
    Nt = max(len(t) for _, t in batch)
    B = len(batch)
    src = torch.full((B, Ns), pad_id, dtype=torch.long)
    tgt = torch.full((B, Nt), pad_id, dtype=torch.long)
    for b, (s, t) in enumerate(batch):
        src[b, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt[b, : len(t)] = torch.tensor(t, dtype=torch.long)
    return {
        "src_ids": src,
        "src_pad_mask": src == pad_id,
        "tgt_ids": tgt,
        "tgt_pad_mask": tgt == pad_id,
    }


class BaselineTransformer(nn.Module):
    """Encoder-decoder Transformer tiêu chuẩn, embedding đầu vào/đầu ra chia sẻ (Mục 4.3.2)."""

    def __init__(self, vocab_size_total: int, cfg: dict, max_len: int = 256):
        super().__init__()
        d_model = cfg["d_model"]
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size_total, d_model, padding_idx=PAD_ID)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=cfg["n_heads"],
            num_encoder_layers=cfg["n_layers"],
            num_decoder_layers=cfg["n_layers"],
            dim_feedforward=cfg["d_ff"],
            dropout=cfg["dropout"],
            activation="relu",  # Transformer gốc dùng ReLU (Mục 2.2.2)
            batch_first=True,
        )
        self.out_proj = nn.Linear(d_model, vocab_size_total)
        self.out_proj.weight = self.tok_emb.weight  # chia sẻ trọng số đầu vào/đầu ra
        # Khởi tạo nhỏ hơn mặc định (N(0,1)) cho embedding dùng chung đầu vào/đầu ra:
        # nếu không, logits đầu ra (chưa scale) có phương sai lớn ngay từ đầu, khiến
        # cross-entropy ban đầu cao hơn nhiều so với ln(|V|) và hội tụ chậm không cần thiết.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=d_model**-0.5)
        with torch.no_grad():
            self.tok_emb.weight[PAD_ID].zero_()

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        x = self.tok_emb(ids) * math.sqrt(self.d_model)
        positions = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_enc(positions)
        return self.dropout(x)

    def forward(self, batch: dict) -> torch.Tensor:
        src = self._embed(batch["src_ids"])
        tgt_in = batch["tgt_ids"][:, :-1]
        tgt = self._embed(tgt_in)
        T = tgt.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt.device)
        out = self.transformer(
            src,
            tgt,
            tgt_mask=causal_mask,
            src_key_padding_mask=batch["src_pad_mask"],
            tgt_key_padding_mask=batch["tgt_pad_mask"][:, :-1],
            memory_key_padding_mask=batch["src_pad_mask"],
        )
        return self.out_proj(out)  # (B,T,V)

    @torch.no_grad()
    def generate_beam(self, src_ids: list[int], bos_id: int, eos_id: int, beam_size: int, max_len: int, device):
        self.eval()
        src_t = torch.tensor([src_ids], device=device)
        memory = self.transformer.encoder(self._embed(src_t))
        beams = [([bos_id], 0.0)]
        for _ in range(max_len):
            new_beams = []
            all_done = True
            for seq, score in beams:
                if seq[-1] == eos_id:
                    new_beams.append((seq, score))
                    continue
                all_done = False
                tgt_t = torch.tensor([seq], device=device)
                causal = nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)
                dec_out = self.transformer.decoder(self._embed(tgt_t), memory, tgt_mask=causal)
                logprobs = torch.log_softmax(self.out_proj(dec_out[0, -1]), dim=-1)
                k = min(beam_size, logprobs.size(0))
                vals, idxs = logprobs.topk(k)
                for v, idx in zip(vals.tolist(), idxs.tolist()):
                    new_beams.append((seq + [idx], score + v))
            new_beams.sort(key=lambda x: -x[1])
            beams = new_beams[:beam_size]
            if all_done:
                break
        best = beams[0][0]
        if best and best[0] == bos_id:
            best = best[1:]
        if best and best[-1] == eos_id:
            best = best[:-1]
        return best
