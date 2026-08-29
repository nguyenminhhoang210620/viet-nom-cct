#!/usr/bin/env bash
# Chạy thử toàn bộ pipeline trên dữ liệu GIẢ LẬP, trên CPU, với vài epoch nhỏ.
# Mục đích: xác nhận pipeline chạy đúng logic, KHÔNG phản ánh chất lượng chuyển tự thật.
set -euo pipefail
cd "$(dirname "$0")/.."

N_SENT="${1:-300}"

echo "== 1/5: sinh dữ liệu giả lập ($N_SENT câu/hệ chữ) =="
python scripts/make_sample_data.py --n_sentences "$N_SENT"
mkdir -p data/raw/han data/raw/nom
cp data/sample/han/*.tsv data/raw/han/
cp data/sample/nom/*.tsv data/raw/nom/

echo "== 2/5: tiền xử lý =="
# python -m src.data_prep --config configs/default.yaml --script han
python -m src.data_prep --config configs/default.yaml --script nom

echo "== 3/5: huấn luyện CCT-C và CCT-D (demo, vài epoch, CPU) =="
python -m src.train --config configs/default.yaml --script han --model_type cct_c --max_epochs 2 --device cpu
python -m src.train --config configs/default.yaml --script han --model_type cct_d --max_epochs 2 --device cpu

echo "== 4/5: đánh giá =="
python -m src.evaluate --config configs/default.yaml --script han --model_type cct_c \
    --checkpoint checkpoints/han/cct_c/best.pt --device cpu
python -m src.evaluate --config configs/default.yaml --script han --model_type cct_d \
    --checkpoint checkpoints/han/cct_d/best.pt --device cpu --beam_size 2

echo "== 5/5: chuyển tự thử một câu (Quốc ngữ -> Hán/Nôm) =="
python -m src.infer --config configs/default.yaml --script han --model_type cct_d \
    --checkpoint checkpoints/han/cct_d/best.pt --device cpu --text "nhất nhị tam đại sơn"

echo "== Xong. Đây là kết quả demo trên dữ liệu giả lập, chỉ để kiểm tra pipeline. =="
