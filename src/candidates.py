"""Xây tập ứng viên C(x_i) cho chiều chuyển tự Quốc ngữ -> Hán/Nôm (Mục 3.2.2).

Mô hình chỉ có MỘT chiều (nguồn = âm tiết Quốc ngữ, đích = ký tự Hán/Nôm), nên tập
ứng viên là một ánh xạ phẳng, xây TRƯỚC khi huấn luyện, từ hai nguồn:
  1. Ngữ liệu song song của tập huấn luyện + phát triển (không dùng tập kiểm tra).
  2. Từ điển Hán/Nôm <-> Quốc ngữ, bổ sung ứng viên cho token hiếm/không có trong ngữ liệu.

Kết quả: candidates[qn_syll] -> list[int]  (id các ký tự Hán/Nôm hợp lệ)

Khoá của ánh xạ là TOKEN NGUỒN DẠNG CHUỖI (không phải id), vì một token nguồn không có
trong vocab (OOV thật sự hiếm) vẫn cần được nhận diện để loại khỏi hàm mất mát và giữ
nguyên ở đầu ra khi suy luận (Mục 3.8.4) — tra bằng id sẽ làm mất phân biệt này do mọi
token lạ đều bị gộp về [UNK].
"""
from __future__ import annotations

import json
from collections import defaultdict

from .vocab import Vocab


def build_candidates(train_dev_pairs, dict_entries, vocab: Vocab) -> tuple[dict, dict]:
    """train_dev_pairs: list[(qn_tokens, hn_tokens)] đã căn chỉnh cùng độ dài.
    dict_entries: list[(hn_char, qn_syllable)].

    Trả về (candidates, stats) — candidates là {qn_syll: [id_hn, ...]} đã sort, lọc theo vocab.
    """
    raw = defaultdict(set)  # nguồn = âm Quốc ngữ, đích = ký tự Hán/Nôm
    for qn_toks, hn_toks in train_dev_pairs:
        for qn, hn in zip(qn_toks, hn_toks):
            raw[qn].add(hn)
    for hn, qn in dict_entries:
        raw[qn].add(hn)

    out = {}
    sizes = []
    singleton = 0
    for src_tok, tgt_set in raw.items():
        ids = sorted({vocab.encode(t) for t in tgt_set if t in vocab.token2id})
        if not ids:
            continue
        out[src_tok] = ids
        sizes.append(len(ids))
        if len(ids) == 1:
            singleton += 1

    if sizes:
        stats = {
            "n_source_tokens": len(sizes),
            "mean_candidates": sum(sizes) / len(sizes),
            "max_candidates": max(sizes),
            "pct_singleton": 100.0 * singleton / len(sizes),
        }
    else:
        stats = {"n_source_tokens": 0, "mean_candidates": 0, "max_candidates": 0, "pct_singleton": 0}
    return out, stats


def save_candidates(candidates: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)


def load_candidates(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lookup(candidates: dict, src_token: str) -> list[int]:
    """Tra C(x_i); trả về [] nếu token nguồn không có ứng viên nào (OOV thật sự)."""
    return candidates.get(src_token, [])
