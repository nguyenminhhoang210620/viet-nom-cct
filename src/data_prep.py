"""Tiền xử lý ngữ liệu: tokenize, lọc, chia tập, xây từ vựng + tập ứng viên.

Tương ứng Mục 4.1.1 (tiền xử lý), Mục 3.2.2 (tập ứng viên) và Bảng 3.3 (từ vựng)
của báo cáo tham khảo.

Cách dùng:
    python -m src.data_prep --config configs/default.yaml --script han
    python -m src.data_prep --config configs/default.yaml --script nom
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .candidates import build_candidates, save_candidates
from .utils import ensure_dir, load_config, read_tsv, set_seed
from .vocab import Vocab

_WS_RE = re.compile(r"\s+")


def tokenize_qn(sent: str) -> list[str]:
    """Câu Quốc ngữ: âm tiết cách nhau bằng khoảng trắng, hạ chữ thường."""
    sent = _WS_RE.sub(" ", sent.strip().lower())
    return sent.split(" ") if sent else []


def tokenize_hn(sent: str) -> list[str]:
    """Câu Hán/Nôm: mỗi ký tự Unicode là một token (bỏ khoảng trắng nếu có)."""
    return list(sent.strip().replace(" ", ""))


def load_and_tokenize_parallel(path: str) -> list[tuple[list[str], list[str]]]:
    pairs = []
    n_raw = 0
    n_dropped_len = 0
    seen = set()
    n_dup = 0
    for qn_raw, hn_raw in read_tsv(path):
        n_raw += 1
        qn_toks = tokenize_qn(qn_raw)
        hn_toks = tokenize_hn(hn_raw)
        key = (tuple(qn_toks), tuple(hn_toks))
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        if not qn_toks or not hn_toks or len(qn_toks) != len(hn_toks):
            n_dropped_len += 1
            continue
        pairs.append((qn_toks, hn_toks))
    print(
        f"[data_prep] {path}: {n_raw} dòng thô -> {len(pairs)} cặp giữ lại "
        f"({n_dup} trùng lặp, {n_dropped_len} lệch số token bị loại)"
    )
    return pairs


def stratified_split(pairs, ratios, seed, n_bins):
    """Chia train/dev/test phân tầng theo độ dài câu (Mục 4.1.1)."""
    rng = random.Random(seed)
    by_len = defaultdict(list)
    for p in pairs:
        by_len[len(p[0])].append(p)

    lengths = sorted(by_len)
    if not lengths:
        return [], [], []
    lo, hi = lengths[0], lengths[-1]
    bin_width = max(1, (hi - lo + 1) // n_bins or 1)

    train, dev, test = [], [], []
    for length, items in by_len.items():
        rng.shuffle(items)
        n = len(items)
        n_dev = max(1, round(n * ratios[1])) if n >= 3 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        n_dev = min(n_dev, n)
        n_test = min(n_test, n - n_dev)
        dev.extend(items[:n_dev])
        test.extend(items[n_dev : n_dev + n_test])
        train.extend(items[n_dev + n_test :])
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


def write_jsonl(pairs, path):
    with open(path, "w", encoding="utf-8") as f:
        for qn_toks, hn_toks in pairs:
            f.write(json.dumps({"qn": qn_toks, "hn": hn_toks}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--script", required=True, choices=["han", "nom"])
    args = ap.parse_args()

    cfg = load_config(args.config, script=args.script)
    set_seed(cfg["seed"])
    ensure_dir(cfg["processed_dir"])

    raw_dir = Path(cfg["raw_dir"])
    pairs = load_and_tokenize_parallel(str(raw_dir / "parallel.tsv"))
    if not pairs:
        raise SystemExit(
            f"Không tìm thấy dữ liệu hợp lệ trong {raw_dir / 'parallel.tsv'}. "
            f"Xem README.md mục 3 để biết định dạng, hoặc chạy scripts/make_sample_data.py trước."
        )

    dict_rows = read_tsv(str(raw_dir / "dict.tsv"))  # (hn_char, qn_syllable)
    print(f"[data_prep] {len(dict_rows)} mục từ điển")

    train, dev, test = stratified_split(pairs, cfg["split_ratios"], cfg["split_seed"], cfg["length_bins"])
    print(f"[data_prep] chia tập: train={len(train)} dev={len(dev)} test={len(test)}")

    # Từ vựng + tập ứng viên: chỉ dùng train + dev (Mục 3.2.2), cộng từ điển.
    train_dev = train + dev
    vi_tokens: set[str] = set()
    no_tokens: set[str] = set()
    for qn_toks, hn_toks in train_dev:
        vi_tokens.update(qn_toks)
        no_tokens.update(hn_toks)
    for hn, qn in dict_rows:
        no_tokens.add(hn)
        vi_tokens.add(qn)

    vocab = Vocab.build(vi_tokens, no_tokens)
    print(f"[data_prep] từ vựng: {len(vocab)} token ({vocab.n_vi} Quốc ngữ, {vocab.n_no} Hán/Nôm)")

    candidates, stats = build_candidates(train_dev, dict_rows, vocab)
    print(
        f"[data_prep] tập ứng viên (vi->hán/nôm): {stats['n_source_tokens']} token nguồn, "
        f"trung bình {stats['mean_candidates']:.2f} ứng viên, lớn nhất {stats['max_candidates']}, "
        f"{stats['pct_singleton']:.1f}% đơn trị"
    )

    out_dir = Path(cfg["processed_dir"])
    vocab.save(str(out_dir / "vocab.json"))
    save_candidates(candidates, str(out_dir / "candidates.json"))
    write_jsonl(train, str(out_dir / "train.jsonl"))
    write_jsonl(dev, str(out_dir / "dev.jsonl"))
    write_jsonl(test, str(out_dir / "test.jsonl"))
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_train": len(train),
                "n_dev": len(dev),
                "n_test": len(test),
                "vocab_size": len(vocab),
                "candidate_stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[data_prep] đã ghi kết quả vào {out_dir}/")


if __name__ == "__main__":
    main()
