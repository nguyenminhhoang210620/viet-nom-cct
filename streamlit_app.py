"""Streamlit app: chuyen tu Quoc ngu -> Han/Nom bang mo hinh CCT da huan luyen.

Chay:
    streamlit run streamlit_app.py

Yeu cau truoc khi chay:
  - Da co checkpoint da huan luyen (vd. tren Kaggle) dat tai
    checkpoints/<script>/<model_type>/best.pt
  - Da co data/processed/<script>/{vocab.json,candidates.json} SINH RA TU CUNG
    lan tien xu ly dung de huan luyen checkpoint do (id token phai khop!).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from src.baseline_transformer import BaselineTransformer
from src.candidates import load_candidates
from src.data_prep import tokenize_qn
from src.infer import prepare_example, reconstruct, run_cct_c
from src.train import build_cct_model
from src.utils import load_config
from src.vocab import Vocab

st.set_page_config(page_title="Chuyen tu Quoc ngu -> Han/Nom", page_icon="📜", layout="centered")

CONFIG_PATH = "configs/default.yaml"


@st.cache_resource(show_spinner="Dang nap mo hinh...")
def load_pipeline(script: str, model_type: str, checkpoint_path: str, device_str: str):
    cfg = load_config(CONFIG_PATH, script=script)
    processed = Path(cfg["processed_dir"])
    vocab_path = processed / "vocab.json"
    cand_path = processed / "candidates.json"
    if not vocab_path.exists() or not cand_path.exists():
        raise FileNotFoundError(
            f"Thieu {vocab_path} hoac {cand_path}. Chay `python -m src.data_prep --config {CONFIG_PATH} "
            f"--script {script}` truoc, dung dung ngu lieu da dung de huan luyen checkpoint nay."
        )
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Khong tim thay checkpoint tai {ckpt_file}. Tai file best.pt tu Output cua notebook "
            f"Kaggle ve va dat dung duong dan nay."
        )

    vocab = Vocab.load(str(vocab_path))
    device = torch.device(device_str)
    ckpt = torch.load(str(ckpt_file), map_location=device)

    if model_type in ("cct_c", "cct_d"):
        candidates = load_candidates(str(cand_path))
        model = build_cct_model(model_type, len(vocab), cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return {"cfg": cfg, "vocab": vocab, "candidates": candidates, "model": model, "device": device}
    else:
        vocab_size_total = len(vocab) + 2
        model = BaselineTransformer(vocab_size_total, cfg["baseline"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return {"cfg": cfg, "vocab": vocab, "candidates": None, "model": model, "device": device}


def translit(pipeline: dict, model_type: str, text: str, beam_size: int):
    cfg, vocab, model, device = pipeline["cfg"], pipeline["vocab"], pipeline["model"], pipeline["device"]

    if model_type in ("cct_c", "cct_d"):
        candidates = pipeline["candidates"]
        src_tokens, src_ids, cand_ids_per_pos, is_oov = prepare_example(text, vocab, candidates)
        if not src_tokens:
            return "", []
        if model_type == "cct_c":
            pred_tokens = run_cct_c(model, vocab, src_ids, cand_ids_per_pos, device)
        else:
            pred_ids = model.generate_beam(src_ids, cand_ids_per_pos, beam_size, device)
            pred_tokens = [vocab.decode(i) for i in pred_ids]
        result = reconstruct(src_tokens, pred_tokens, is_oov)
        rows = [
            {
                "Am tiet": src_tokens[i],
                "Ky tu dich": src_tokens[i] if is_oov[i] else pred_tokens[i],
                "So ung vien": len(cand_ids_per_pos[i]),
                "OOV": is_oov[i],
            }
            for i in range(len(src_tokens))
        ]
        return result, rows
    else:
        bos_id, eos_id = len(vocab), len(vocab) + 1
        src_tokens = tokenize_qn(text)
        if not src_tokens:
            return "", []
        src_ids = [vocab.encode(t) for t in src_tokens]
        pred_ids = model.generate_beam(
            src_ids, bos_id, eos_id, beam_size, cfg["encoder"]["max_len"], device
        )
        pred_tokens = [vocab.decode(i) for i in pred_ids]
        result = "".join(pred_tokens)
        rows = [{"Am tiet": t, "Ky tu dich": "?", "So ung vien": "-", "OOV": "-"} for t in src_tokens]
        return result, rows


# ---------------- Giao dien ----------------
st.title("📜 Chuyen tu Quoc ngu → Han/Nom")
st.caption(
    "Cong cu chuyen tu (khong phai dich nghia) dua tren mo hinh CCT-C / CCT-D — "
    "xem 25D2KLTTNT03_BaoCao.pdf de biet phuong phap."
)

with st.sidebar:
    st.header("Cau hinh mo hinh")
    script = st.selectbox("He chu", ["nom", "han"], help="Moi he chu co mot mo hinh rieng.")
    model_type = st.selectbox("Kien truc", ["cct_d", "cct_c", "baseline"], index=0)
    default_ckpt = f"checkpoints/{script}/{model_type}/best.pt"
    checkpoint_path = st.text_input("Duong dan checkpoint", value=default_ckpt)
    device_str = st.selectbox("Thiet bi", ["cpu", "cuda"], index=0)
    beam_size = st.slider("Beam size (CCT-D / baseline)", 1, 10, 5)
    load_clicked = st.button("Nap mo hinh", type="primary")

if "pipeline_key" not in st.session_state:
    st.session_state.pipeline_key = None
if load_clicked:
    st.session_state.pipeline_key = (script, model_type, checkpoint_path, device_str)

pipeline = None
if st.session_state.pipeline_key:
    s, mt, ck, dv = st.session_state.pipeline_key
    try:
        pipeline = load_pipeline(s, mt, ck, dv)
        st.success(f"Da nap mo hinh {mt.upper()} — he chu {s} — tu {ck}")
    except Exception as e:
        st.error(str(e))
else:
    st.info("Chon cau hinh o thanh ben roi bam **Nap mo hinh**.")

text = st.text_input(
    "Cau Quoc ngu (cac am tiet cach nhau bang khoang trang)",
    value="nam quoc son ha nam de cu",
)

if st.button("Chuyen tu"):
    if pipeline is None:
        st.warning("Nap mo hinh o thanh ben truoc da.")
    elif not text.strip():
        st.warning("Nhap mot cau truoc da.")
    else:
        with st.spinner("Dang chuyen tu..."):
            result, rows = translit(pipeline, st.session_state.pipeline_key[1], text, beam_size)
        st.subheader("Ket qua")
        st.markdown(f"## {result}")
        if rows and st.session_state.pipeline_key[1] != "baseline":
            with st.expander("Chi tiet theo tung am tiet"):
                st.table(rows)
            if any(r.get("OOV") for r in rows):
                st.caption("⚠️ Cac am tiet danh dau OOV khong co trong tap ung vien — giu nguyen o dau ra.")
