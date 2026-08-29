"""CLI chuyển tự một câu Quốc ngữ sang Hán/Nôm, dùng checkpoint đã huấn luyện.

Cách dùng:
    python -m src.infer --config configs/default.yaml --script han --model_type cct_d \
        --checkpoint checkpoints/han/cct_d/best.pt --text "nam quốc sơn hà"

--text là câu Quốc ngữ, các âm tiết cách nhau bằng dấu cách. Mô hình chỉ chuyển tự một
chiều Quốc ngữ -> Hán/Nôm.

Token nguồn không có ứng viên nào (OOV thật sự, Mục 3.8.4) được giữ nguyên ở đầu ra
thay vì bị mô hình đoán bừa — chỉ áp dụng cho CCT-C/CCT-D (baseline không có cơ chế
ràng buộc ứng viên nên không thực hiện bước này).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .baseline_transformer import BaselineTransformer
from .candidates import load_candidates, lookup
from .data_prep import tokenize_qn
from .train import build_cct_model, resolve_device
from .utils import load_config
from .vocab import PAD_ID, UNK, Vocab


def prepare_example(text: str, vocab: Vocab, candidates: dict):
    src_tokens = tokenize_qn(text)
    src_ids, cand_ids_per_pos, is_oov = [], [], []
    for tok in src_tokens:
        src_ids.append(vocab.encode(tok))
        cands = lookup(candidates, tok)
        if not cands:
            cands = [vocab.token2id.get(UNK, 1)]
            is_oov.append(True)
        else:
            is_oov.append(False)
        cand_ids_per_pos.append(cands)
    return src_tokens, src_ids, cand_ids_per_pos, is_oov


def reconstruct(src_tokens, pred_tokens, is_oov) -> str:
    out = [src_tokens[i] if is_oov[i] else pred_tokens[i] for i in range(len(src_tokens))]
    return "".join(out)


def run_cct_c(model, vocab, src_ids, cand_ids_per_pos, device):
    n = len(src_ids)
    c_max = max(len(c) for c in cand_ids_per_pos)
    batch = {
        "src_ids": torch.tensor([src_ids], device=device),
        "content_mask": torch.ones((1, n), dtype=torch.bool, device=device),
        "cand_ids": torch.full((1, n, c_max), PAD_ID, dtype=torch.long, device=device),
        "cand_mask": torch.zeros((1, n, c_max), dtype=torch.bool, device=device),
    }
    for i, cands in enumerate(cand_ids_per_pos):
        batch["cand_ids"][0, i, : len(cands)] = torch.tensor(cands, device=device)
        batch["cand_mask"][0, i, : len(cands)] = True
    pred_ids = model.predict(batch)[0].tolist()
    return [vocab.decode(i) for i in pred_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--script", required=True, choices=["han", "nom"])
    ap.add_argument("--model_type", required=True, choices=["cct_c", "cct_d", "baseline"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--beam_size", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, script=args.script)
    device = resolve_device(args, cfg)
    processed = Path(cfg["processed_dir"])
    vocab = Vocab.load(str(processed / "vocab.json"))
    ckpt = torch.load(args.checkpoint, map_location=device)

    if args.model_type in ("cct_c", "cct_d"):
        candidates = load_candidates(str(processed / "candidates.json"))
        src_tokens, src_ids, cand_ids_per_pos, is_oov = prepare_example(args.text, vocab, candidates)
        if not src_tokens:
            print("(chuỗi rỗng sau khi tách token)")
            return
        model = build_cct_model(args.model_type, len(vocab), cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        if args.model_type == "cct_c":
            pred_tokens = run_cct_c(model, vocab, src_ids, cand_ids_per_pos, device)
        else:
            pred_ids = model.generate_beam(
                src_ids, cand_ids_per_pos, args.beam_size or cfg["decoder"]["beam_size"], device
            )
            pred_tokens = [vocab.decode(i) for i in pred_ids]
        result = reconstruct(src_tokens, pred_tokens, is_oov)
    else:
        bos_id, eos_id = len(vocab), len(vocab) + 1
        vocab_size_total = len(vocab) + 2
        src_tokens = tokenize_qn(args.text)
        src_ids = [vocab.encode(t) for t in src_tokens]
        model = BaselineTransformer(vocab_size_total, cfg["baseline"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        pred_ids = model.generate_beam(
            src_ids, bos_id, eos_id, args.beam_size or cfg["baseline"]["beam_size"], cfg["encoder"]["max_len"], device
        )
        pred_tokens = [vocab.decode(i) for i in pred_ids]
        result = "".join(pred_tokens)

    print(f"Nguồn (Quốc ngữ): {args.text}")
    print(f"Kết quả (Hán/Nôm): {result}")


if __name__ == "__main__":
    main()
