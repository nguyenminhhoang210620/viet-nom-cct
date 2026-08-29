"""Từ vựng thống nhất Quốc ngữ + Hán/Nôm.

Mô hình chỉ chuyển tự MỘT CHIỀU: Quốc ngữ (nguồn) -> Hán/Nôm (đích), theo đúng phạm vi
của báo cáo (không có chiều ngược lại no2vi, nên không cần token chỉ thị chiều).

Cấu trúc:
    0            [PAD]
    1            [UNK]
    2 .. 1+V     các token tiếng Việt / Quốc ngữ (âm tiết, phía nguồn)
    2+V .. 1+V+N  các token Hán/Nôm (ký tự, phía đích)

Từ vựng được xây một lần cho mỗi hệ chữ (Hán hoặc Nôm) từ tập huấn luyện + phát triển
của ngữ liệu song song, cộng thêm từ điển — KHÔNG dùng tập kiểm tra (Mục 3.2.2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

PAD, UNK = "[PAD]", "[UNK]"
SPECIAL_TOKENS = [PAD, UNK]
PAD_ID, UNK_ID = 0, 1


@dataclass
class Vocab:
    token2id: dict = field(default_factory=dict)
    id2token: list = field(default_factory=list)
    n_vi: int = 0
    n_no: int = 0

    @classmethod
    def build(cls, vi_tokens: Iterable[str], no_tokens: Iterable[str]) -> "Vocab":
        vi_sorted = sorted(set(vi_tokens) - set(SPECIAL_TOKENS))
        no_sorted = sorted(set(no_tokens) - set(SPECIAL_TOKENS))
        id2token = list(SPECIAL_TOKENS) + vi_sorted + no_sorted
        token2id = {t: i for i, t in enumerate(id2token)}
        return cls(token2id=token2id, id2token=id2token, n_vi=len(vi_sorted), n_no=len(no_sorted))

    def __len__(self) -> int:
        return len(self.id2token)

    def encode(self, token: str) -> int:
        return self.token2id.get(token, UNK_ID)

    def encode_seq(self, tokens: Iterable[str]) -> list[int]:
        return [self.encode(t) for t in tokens]

    def decode(self, idx: int) -> str:
        if 0 <= idx < len(self.id2token):
            return self.id2token[idx]
        return UNK

    def is_vi_id(self, idx: int) -> bool:
        lo = len(SPECIAL_TOKENS)
        return lo <= idx < lo + self.n_vi

    def is_no_id(self, idx: int) -> bool:
        lo = len(SPECIAL_TOKENS) + self.n_vi
        return lo <= idx < lo + self.n_no

    def target_vocab_ids(self) -> list[int]:
        """Toàn bộ id thuộc phía từ vựng đích (Hán/Nôm) — dùng khi debug / baseline."""
        return [i for i in range(len(self)) if self.is_no_id(i)]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"id2token": self.id2token, "n_vi": self.n_vi, "n_no": self.n_no},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        id2token = data["id2token"]
        token2id = {t: i for i, t in enumerate(id2token)}
        return cls(token2id=token2id, id2token=id2token, n_vi=data["n_vi"], n_no=data["n_no"])
