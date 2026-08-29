"""Đánh giá trên tập kiểm tra: BLEU-4 + độ chính xác theo vị trí (Mục 4.2).

Mô hình chỉ có một chiều chuyển tự (Quốc ngữ -> Hán/Nôm), nên kết quả không cần tách
theo chiều như trước — chỉ còn tách theo hệ chữ {Hán, Nôm} (chạy script này riêng cho
từng --script).

Cách dùng:
    python -m src.evaluate --config configs/default.yaml --script han --model_type cct_d \
        --checkpoint checkpoints/han/cct_d/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sacrebleu
import torch
from torch.utils.data import DataLoader

from .baseline_transformer import BaselineDataset, BaselineTransformer, collate_baseline
from .candidates import load_candidates
from .dataset import CCTDataset, collate_cct
from .train import build_cct_model, resolve_device
from .utils import load_config
from .vocab import Vocab


def bleu(hyps: list[str], refs: list[str]) -> float:
    if not hyps:
        return 0.0
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none").score


def eval_cct_c(model, test_ds, vocab, device, batch_size=64):
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_cct)
    result = {"hyp": [], "ref": [], "correct": 0, "valid": 0}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred_ids = model.predict(batch_dev).cpu()
            for b in range(len(batch["src_tokens"])):
                n = len(batch["src_tokens"][b])
                pred_toks = [vocab.decode(int(pred_ids[b, i])) for i in range(n)]
                ref_toks = batch["tgt_tokens"][b]
                result["hyp"].append(" ".join(pred_toks))
                result["ref"].append(" ".join(ref_toks))
                local_idx = batch["local_idx"][b]
                for i in range(n):
                    if local_idx[i].item() == -100:
                        continue
                    result["valid"] += 1
                    if pred_toks[i] == ref_toks[i]:
                        result["correct"] += 1
    return result


def eval_cct_d(model, test_ds, vocab, device, beam_size):
    result = {"hyp": [], "ref": [], "correct": 0, "valid": 0}
    model.eval()
    for ex in test_ds.examples:
        pred_ids = model.generate_beam(
            src_ids=ex.src_ids,
            cand_ids_per_pos=ex.cand_ids_list,
            beam_size=beam_size,
            device=device,
        )
        pred_toks = [vocab.decode(i) for i in pred_ids]
        ref_toks = ex.tgt_tokens
        result["hyp"].append(" ".join(pred_toks))
        result["ref"].append(" ".join(ref_toks))
        for i, li in enumerate(ex.local_idx):
            if li == -100:
                continue
            result["valid"] += 1
            if i < len(pred_toks) and pred_toks[i] == ref_toks[i]:
                result["correct"] += 1
    return result


def eval_baseline(model, test_ds, vocab, bos_id, eos_id, device, beam_size, max_len):
    result = {"hyp": [], "ref": []}
    model.eval()
    for src_toks, tgt_toks in test_ds.examples:
        src_ids = [vocab.encode(t) for t in src_toks]
        pred_ids = model.generate_beam(src_ids, bos_id, eos_id, beam_size, max_len, device)
        pred_toks = [vocab.decode(i) for i in pred_ids]
        result["hyp"].append(" ".join(pred_toks))
        result["ref"].append(" ".join(tgt_toks))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--script", required=True, choices=["han", "nom"])
    ap.add_argument("--model_type", required=True, choices=["cct_c", "cct_d", "baseline"])
    ap.add_argument("--checkpoint", required=True)
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
        test_ds = CCTDataset(str(processed / "test.jsonl"), vocab, candidates, max_len=cfg["encoder"]["max_len"])
        model = build_cct_model(args.model_type, len(vocab), cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        if args.model_type == "cct_c":
            result = eval_cct_c(model, test_ds, vocab, device)
        else:
            result = eval_cct_d(model, test_ds, vocab, device, args.beam_size or cfg["decoder"]["beam_size"])
        b = bleu(result["hyp"], result["ref"])
        acc = 100.0 * result["correct"] / max(1, result["valid"])
        print(f"\n=== Kết quả {args.model_type.upper()} trên {cfg['script']} (tập kiểm tra, vi->hán/nôm) ===")
        print(f"  BLEU={b:.2f}  token_acc={acc:.2f}%  (n={len(result['hyp'])})")
    else:
        bos_id, eos_id = len(vocab), len(vocab) + 1
        vocab_size_total = len(vocab) + 2
        test_ds = BaselineDataset(str(processed / "test.jsonl"), vocab, bos_id, eos_id)
        model = BaselineTransformer(vocab_size_total, cfg["baseline"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        result = eval_baseline(
            model, test_ds, vocab, bos_id, eos_id, device,
            args.beam_size or cfg["baseline"]["beam_size"], max_len=cfg["encoder"]["max_len"],
        )
        b = bleu(result["hyp"], result["ref"])
        print(f"\n=== Kết quả baseline Transformer trên {cfg['script']} (tập kiểm tra, vi->hán/nôm) ===")
        print(f"  BLEU={b:.2f}  (n={len(result['hyp'])})")


if __name__ == "__main__":
    main()
