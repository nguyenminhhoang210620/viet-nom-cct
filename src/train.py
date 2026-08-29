"""Vòng lặp huấn luyện cho CCT-C, CCT-D và Transformer cơ sở (Bảng 4.3).

Cách dùng:
    python -m src.train --config configs/default.yaml --script han --model_type cct_c
    python -m src.train --config configs/default.yaml --script han --model_type cct_d
    python -m src.train --config configs/default.yaml --script han --model_type baseline

Thêm --max_epochs / --device / --batch_size để ghi đè nhanh khi smoke-test trên CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .baseline_transformer import BOS, EOS, BaselineDataset, BaselineTransformer, collate_baseline
from .candidates import load_candidates
from .cct_c import CCTCModel
from .cct_d import CCTDModel
from .dataset import CCTDataset, collate_cct
from .losses import cct_masked_loss, token_accuracy
from .utils import ensure_dir, load_config, set_seed
from .vocab import PAD_ID, Vocab


def cosine_warmup(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def inv_sqrt_warmup(step, warmup_steps):
    step = max(1, step)
    return min(step**-0.5, step * warmup_steps**-1.5) * (warmup_steps**0.5)


def resolve_device(args, cfg: dict) -> torch.device:
    requested = args.device or cfg["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA không khả dụng, chuyển sang CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_cct_model(model_type: str, vocab_size: int, cfg: dict):
    if model_type == "cct_c":
        return CCTCModel(vocab_size, cfg["encoder"], cfg["cct_c"])
    if model_type == "cct_d":
        return CCTDModel(vocab_size, cfg["encoder"], cfg["decoder"])
    raise ValueError(model_type)


@torch.no_grad()
def evaluate_cct_dev(model, loader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    total_correct, total_valid = 0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        scores = model(batch)
        loss = cct_masked_loss(scores, batch["local_idx"])
        total_loss += loss.item()
        n_batches += 1
        c, v = token_accuracy(scores, batch["local_idx"])
        total_correct += c
        total_valid += v
    model.train()
    avg_loss = total_loss / max(1, n_batches)
    acc = total_correct / max(1, total_valid)
    return avg_loss, acc


def train_cct(args, cfg):
    device = resolve_device(args, cfg)
    processed = Path(cfg["processed_dir"])
    vocab = Vocab.load(str(processed / "vocab.json"))
    candidates = load_candidates(str(processed / "candidates.json"))

    train_ds = CCTDataset(str(processed / "train.jsonl"), vocab, candidates, max_len=cfg["encoder"]["max_len"])
    dev_ds = CCTDataset(str(processed / "dev.jsonl"), vocab, candidates, max_len=cfg["encoder"]["max_len"])
    bs = args.batch_size or cfg["train_cct"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_cct)
    dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False, collate_fn=collate_cct)
    print(f"[train] {len(train_ds)} mẫu train, {len(dev_ds)} mẫu dev, batch_size={bs}, thiết bị={device}")

    model = build_cct_model(args.model_type, len(vocab), cfg).to(device)
    tcfg = cfg["train_cct"]
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    max_epochs = args.max_epochs or tcfg["max_epochs"]
    accum = tcfg["grad_accum_steps"]
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = max(1, int(total_steps * tcfg["warmup_ratio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: cosine_warmup(s, total_steps, warmup_steps))

    ckpt_dir = Path(cfg["checkpoint_dir"]) / args.model_type
    ensure_dir(str(ckpt_dir))
    best_dev_loss = float("inf")
    patience = tcfg["early_stop_patience"]
    bad_evals = 0
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        optim.zero_grad()
        for i, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            scores = model(batch)
            loss = cct_masked_loss(scores, batch["local_idx"]) / accum
            loss.backward()
            running_loss += loss.item() * accum
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                if tcfg["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                optim.step()
                scheduler.step()
                optim.zero_grad()
                global_step += 1

        dev_loss, dev_acc = evaluate_cct_dev(model, dev_loader, device)
        dt = time.time() - t0
        print(
            f"[train][{args.model_type}] epoch {epoch}/{max_epochs} "
            f"train_loss={running_loss / len(train_loader):.4f} dev_loss={dev_loss:.4f} "
            f"dev_token_acc={dev_acc * 100:.2f}% ({dt:.1f}s)"
        )

        torch.save({"model_state": model.state_dict(), "vocab_size": len(vocab)}, ckpt_dir / "last.pt")
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            bad_evals = 0
            torch.save({"model_state": model.state_dict(), "vocab_size": len(vocab)}, ckpt_dir / "best.pt")
            print(f"[train] -> lưu checkpoint tốt nhất (dev_loss={dev_loss:.4f})")
        else:
            bad_evals += 1
            if bad_evals >= patience:
                print(f"[train] dừng sớm sau {patience} lần đánh giá không cải thiện")
                break

    print(f"[train] hoàn tất. Checkpoint tốt nhất: {ckpt_dir / 'best.pt'}")


def train_baseline(args, cfg):
    device = resolve_device(args, cfg)
    processed = Path(cfg["processed_dir"])
    vocab = Vocab.load(str(processed / "vocab.json"))
    bos_id, eos_id = len(vocab), len(vocab) + 1
    vocab_size_total = len(vocab) + 2

    train_ds = BaselineDataset(str(processed / "train.jsonl"), vocab, bos_id, eos_id)
    dev_ds = BaselineDataset(str(processed / "dev.jsonl"), vocab, bos_id, eos_id)
    bs = args.batch_size or 32
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_baseline)
    dev_loader = DataLoader(dev_ds, batch_size=bs, shuffle=False, collate_fn=collate_baseline)
    print(f"[train] baseline: {len(train_ds)} mẫu train, {len(dev_ds)} mẫu dev, thiết bị={device}")

    model = BaselineTransformer(vocab_size_total, cfg["baseline"]).to(device)
    tcfg = cfg["train_baseline"]
    optim = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], betas=(0.9, 0.98), weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: inv_sqrt_warmup(s, tcfg["warmup_steps"]))

    max_epochs = args.max_epochs or tcfg["max_epochs"]
    ckpt_dir = Path(cfg["checkpoint_dir"]) / "baseline"
    ensure_dir(str(ckpt_dir))
    best_dev_loss = float("inf")
    bad_evals = 0
    ls = cfg["baseline"]["label_smoothing"]

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)  # (B,T,V), T = len(tgt)-1
            gold = batch["tgt_ids"][:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), gold.reshape(-1), ignore_index=PAD_ID, label_smoothing=ls
            )
            optim.zero_grad()
            loss.backward()
            if tcfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optim.step()
            scheduler.step()
            running_loss += loss.item()

        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch)
                gold = batch["tgt_ids"][:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1), ignore_index=PAD_ID)
                dev_loss += loss.item()
        dev_loss /= max(1, len(dev_loader))
        dt = time.time() - t0
        print(f"[train][baseline] epoch {epoch}/{max_epochs} train_loss={running_loss / len(train_loader):.4f} dev_loss={dev_loss:.4f} ({dt:.1f}s)")

        torch.save({"model_state": model.state_dict(), "vocab_size_total": vocab_size_total}, ckpt_dir / "last.pt")
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            bad_evals = 0
            torch.save({"model_state": model.state_dict(), "vocab_size_total": vocab_size_total}, ckpt_dir / "best.pt")
            print(f"[train] -> lưu checkpoint tốt nhất (dev_loss={dev_loss:.4f})")
        else:
            bad_evals += 1
            if bad_evals >= tcfg["early_stop_patience"]:
                print(f"[train] dừng sớm sau {tcfg['early_stop_patience']} lần đánh giá không cải thiện")
                break

    print(f"[train] hoàn tất. Checkpoint tốt nhất: {ckpt_dir / 'best.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--script", required=True, choices=["han", "nom"])
    ap.add_argument("--model_type", required=True, choices=["cct_c", "cct_d", "baseline"])
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, script=args.script)
    set_seed(cfg["seed"])

    if args.model_type == "baseline":
        train_baseline(args, cfg)
    else:
        train_cct(args, cfg)


if __name__ == "__main__":
    main()
