"""Sinh dữ liệu GIẢ LẬP nhỏ để chạy thử toàn bộ pipeline trước khi có ngữ liệu thật.

QUAN TRỌNG: đây KHÔNG phải ngữ liệu Hán Nôm thật. Bộ chữ Hán dùng các ký tự và âm Hán
Việt phổ biến, có thật (bao gồm vài chữ đa âm thật: 行/樂/本, xem Mục 2.1.1 báo cáo) để
minh hoạ đúng hiện tượng đa trị. Bộ dữ liệu "Nôm" chỉ là một tập con khác của CÙNG kho
chữ Hán Việt này (đổi seed/tỉ lệ), dùng để kiểm tra rằng pipeline xử lý hai hệ chữ độc
lập đúng cách — KHÔNG phải chữ Nôm thật (không có sẵn tự điển Nôm đáng tin cậy trong
môi trường này). Khi có ngữ liệu thật, thay trực tiếp các file trong data/raw/{han,nom}/
theo đúng định dạng mô tả ở README.md.

Cách dùng:
    python scripts/make_sample_data.py --n_sentences 1200
    python scripts/make_sample_data.py --n_sentences 200   # nhanh hơn, để smoke-test
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

# (ký tự Hán, [các âm Hán Việt hợp lệ]) — bao gồm 3 ký tự đa âm thật (行, 樂, 本)
LEXICON = [
    ("一", ["nhất"]), ("二", ["nhị"]), ("三", ["tam"]), ("四", ["tứ"]), ("五", ["ngũ"]),
    ("六", ["lục"]), ("七", ["thất"]), ("八", ["bát"]), ("九", ["cửu"]), ("十", ["thập"]),
    ("人", ["nhân"]), ("大", ["đại"]), ("小", ["tiểu"]), ("山", ["sơn"]), ("水", ["thủy"]),
    ("天", ["thiên"]), ("地", ["địa"]), ("子", ["tử"]), ("女", ["nữ"]), ("王", ["vương"]),
    ("國", ["quốc"]), ("行", ["hành", "hàng", "hạnh"]), ("樂", ["nhạc", "lạc", "nhạo"]),
    ("本", ["bổn", "vốn"]), ("可", ["khả"]), ("不", ["bất"]), ("中", ["trung"]), ("文", ["văn"]),
    ("字", ["tự"]), ("書", ["thư"]), ("學", ["học"]), ("生", ["sinh"]), ("好", ["hảo"]),
    ("高", ["cao"]), ("上", ["thượng"]), ("下", ["hạ"]), ("出", ["xuất"]), ("入", ["nhập"]),
]

PHRASES = [
    (("thượng", "đế"), "上帝"),
    (("nhân", "sinh"), "人生"),
    (("sơn", "thủy"), "山水"),
]


def make_split(lexicon, n_sentences, min_len, max_len, seed):
    rng = random.Random(seed)
    rows_parallel, dict_rows = [], []
    for hn_char, readings in lexicon:
        for r in readings:
            dict_rows.append((hn_char, r))

    seen = set()
    while len(rows_parallel) < n_sentences:
        length = rng.randint(min_len, max_len)
        chosen = [rng.choice(lexicon) for _ in range(length)]
        hn_toks = [c[0] for c in chosen]
        qn_toks = [rng.choice(c[1]) for c in chosen]
        key = tuple(hn_toks)
        if key in seen:
            continue
        seen.add(key)
        rows_parallel.append((" ".join(qn_toks), "".join(hn_toks)))
    return rows_parallel, dict_rows


def write_tsv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a, b in rows:
            f.write(f"{a}\t{b}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sentences", type=int, default=1200)
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=10)
    ap.add_argument("--out_dir", default="data/sample")
    args = ap.parse_args()

    # "han": toàn bộ từ vựng, seed 1
    han_parallel, han_dict = make_split(LEXICON, args.n_sentences, args.min_len, args.max_len, seed=1)
    write_tsv(han_parallel, f"{args.out_dir}/han/parallel.tsv")
    write_tsv(han_dict, f"{args.out_dir}/han/dict.tsv")
    write_tsv([(" ".join(p), h) for p, h in PHRASES], f"{args.out_dir}/han/phrases.tsv")

    # "nom": tập con khác + seed khác, CHỈ để kiểm tra pipeline tách 2 hệ chữ, không phải Nôm thật
    nom_lexicon = LEXICON[5:] + LEXICON[:5]
    nom_parallel, nom_dict = make_split(
        nom_lexicon, max(200, args.n_sentences // 2), args.min_len, args.max_len, seed=2
    )
    write_tsv(nom_parallel, f"{args.out_dir}/nom/parallel.tsv")
    write_tsv(nom_dict, f"{args.out_dir}/nom/dict.tsv")

    print(f"[make_sample_data] han: {len(han_parallel)} câu, {len(han_dict)} mục từ điển -> {args.out_dir}/han/")
    print(f"[make_sample_data] nom: {len(nom_parallel)} câu, {len(nom_dict)} mục từ điển -> {args.out_dir}/nom/")
    print("[make_sample_data] LƯU Ý: dữ liệu giả lập, không phải ngữ liệu Hán Nôm thật.")


if __name__ == "__main__":
    main()
