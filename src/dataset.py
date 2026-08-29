"""PyTorch Dataset + collate cho CCT-C / CCT-D (Mục 3.2.3, ràng buộc tập ứng viên).

Mô hình chỉ chuyển tự MỘT CHIỀU: Quốc ngữ (nguồn) -> Hán/Nôm (đích). Mỗi câu song song
sinh ra đúng MỘT mẫu huấn luyện theo chiều này (không còn chiều no2vi ngược lại).
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset

from .candidates import lookup
from .vocab import PAD_ID, Vocab

IGNORE_INDEX = -100


class CCTExample:
    __slots__ = ("src_ids", "tgt_ids", "cand_ids_list", "local_idx", "src_tokens", "tgt_tokens")

    def __init__(self, src_ids, tgt_ids, cand_ids_list, local_idx, src_tokens, tgt_tokens):
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids
        self.cand_ids_list = cand_ids_list  # list[list[int]], mỗi vị trí 1 danh sách ứng viên (>=1)
        self.local_idx = local_idx  # list[int], IGNORE_INDEX nếu vị trí không hợp lệ
        self.src_tokens = src_tokens  # để phục hồi OOV khi suy luận / debug
        self.tgt_tokens = tgt_tokens


def _build_example(src_tokens, tgt_tokens, vocab: Vocab, candidates: dict) -> CCTExample:
    src_ids, tgt_ids, cand_lists, local_idx = [], [], [], []
    for s_tok, t_tok in zip(src_tokens, tgt_tokens):
        s_id = vocab.encode(s_tok)
        t_id = vocab.encode(t_tok)
        cands = lookup(candidates, s_tok)  # list[int], có thể rỗng
        src_ids.append(s_id)
        tgt_ids.append(t_id)
        if cands and t_id in cands:
            cand_lists.append(cands)
            local_idx.append(cands.index(t_id))
        elif cands:
            # đích đúng không nằm trong tập ứng viên (hiếm, ví dụ do lệch tự điển) -> loại khỏi loss
            cand_lists.append(cands)
            local_idx.append(IGNORE_INDEX)
        else:
            # OOV thật sự: không có ứng viên nào -> giữ 1 ô giả [UNK] để không phá vỡ shape
            cand_lists.append([vocab.token2id.get("[UNK]", 1)])
            local_idx.append(IGNORE_INDEX)
    return CCTExample(src_ids, tgt_ids, cand_lists, local_idx, list(src_tokens), list(tgt_tokens))


class CCTDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab: Vocab, candidates: dict, max_len: int = 128):
        self.examples: list[CCTExample] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                qn_toks, hn_toks = row["qn"], row["hn"]
                if len(qn_toks) == 0 or len(qn_toks) > max_len:
                    continue
                self.examples.append(_build_example(qn_toks, hn_toks, vocab, candidates))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_cct(batch: list[CCTExample]) -> dict:
    B = len(batch)
    n_max = max(len(ex.src_ids) for ex in batch)
    c_max = max(max(len(c) for c in ex.cand_ids_list) for ex in batch)

    src_ids = torch.full((B, n_max), PAD_ID, dtype=torch.long)
    tgt_ids = torch.full((B, n_max), PAD_ID, dtype=torch.long)
    content_mask = torch.zeros((B, n_max), dtype=torch.bool)  # True = vị trí thật (không phải đệm)
    cand_ids = torch.full((B, n_max, c_max), PAD_ID, dtype=torch.long)
    cand_mask = torch.zeros((B, n_max, c_max), dtype=torch.bool)  # True = ô ứng viên hợp lệ
    local_idx = torch.full((B, n_max), IGNORE_INDEX, dtype=torch.long)

    for b, ex in enumerate(batch):
        n = len(ex.src_ids)
        src_ids[b, :n] = torch.tensor(ex.src_ids, dtype=torch.long)
        tgt_ids[b, :n] = torch.tensor(ex.tgt_ids, dtype=torch.long)
        content_mask[b, :n] = True
        for i, cands in enumerate(ex.cand_ids_list):
            cand_ids[b, i, : len(cands)] = torch.tensor(cands, dtype=torch.long)
            cand_mask[b, i, : len(cands)] = True
        local_idx[b, :n] = torch.tensor(ex.local_idx, dtype=torch.long)

    return {
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "content_mask": content_mask,
        "cand_ids": cand_ids,
        "cand_mask": cand_mask,
        "local_idx": local_idx,
        "src_tokens": [ex.src_tokens for ex in batch],
        "tgt_tokens": [ex.tgt_tokens for ex in batch],
    }
