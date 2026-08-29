# viet-nom-cct

Cài đặt tham khảo cho họ mô hình **Candidate-Constrained Transliteration (CCT)** — chuyển tự
hai chiều giữa chữ Quốc ngữ và chữ Hán/Nôm — dựa trên khóa luận tốt nghiệp *"Nghiên cứu mô hình
dịch máy hai chiều giữa chữ Hán Nôm và chữ Quốc ngữ"* (Tống Trọng Tâm, ĐH KHTN – ĐHQG TP.HCM, 2026).

Đây là **bài toán chuyển tự (transliteration/dịch âm)**, không phải dịch nghĩa: mục tiêu là bảo
toàn cách đọc khi chuyển đổi giữa tự dạng Hán/Nôm và chữ Quốc ngữ, không sinh ra một bản dịch nghĩa
sang tiếng Việt hiện đại.

## 1. Cài đặt

```bash
micromamba create --name nlp python=3.11 pip -y
pip install -r requirements.txt
```

Khuyến nghị chạy huấn luyện trên GPU (báo cáo dùng Kaggle Colab GPU T4x2). CPU chỉ đủ để
smoke-test với dữ liệu mẫu.

## 2. Cấu trúc thư mục

```
viet-nom-cct/
├── configs/default.yaml       # siêu tham số, tham khảo Bảng 4.2 / 4.3 báo cáo
├── data/
│   ├── raw/{han,nom}/         #  đặt ngữ liệu THẬT vào đây (xem định dạng bên dưới)
│   ├── sample/{han,nom}/      # dữ liệu giả lập để chạy thử (do make_sample_data.py sinh ra)
│   └── processed/             # output của prepare_data.py: vocab, candidates, splits
├── src/
│   ├── vocab.py                # xây từ vựng thống nhất (Bảng 3.3)
│   ├── candidates.py           # xây tập ứng viên C(x_i) (Mục 3.2.2)
│   ├── data_prep.py            # tokenize + chia train/dev/test (Mục 4.1.1)
│   ├── dataset.py               # PyTorch Dataset/collate có đệm tập ứng viên
│   ├── positional.py            # positional encoding hình sin (Eq 2.1)
│   ├── encoder.py                # bộ mã hoá BERT Post-LN huấn luyện từ đầu (Mục 2.3, 3.4.2)
│   ├── cct_c.py                  # CCT-C: phân loại cục bộ + giải mã song song (Mục 3.5)
│   ├── cct_d.py                  # CCT-D: giải mã Transformer Pre-LN + beam search (Mục 3.6)
│   ├── baseline_transformer.py   # Transformer cơ sở không ràng buộc (Mục 4.3.1)
│   ├── losses.py                  # cross-entropy có ràng buộc ứng viên (Eq 3.16-3.19)
│   ├── train.py                   # vòng lặp huấn luyện (dùng cho cả CCT-C/CCT-D/baseline)
│   ├── evaluate.py                # BLEU-4 + độ chính xác theo vị trí (Mục 4.2)
│   └── infer.py                   # CLI chuyển tự một câu / một file
├── scripts/
│   ├── make_sample_data.py        # sinh dữ liệu giả lập nhỏ
│   └── run_demo.sh                # chạy toàn bộ pipeline từ đầu đến cuối trên dữ liệu mẫu
└── requirements.txt
```

## 3. Định dạng dữ liệu THẬT

Đặt vào `data/raw/han/` (và tương tự `data/raw/nom/`) ba file:

### `parallel.tsv` — ngữ liệu song song (bắt buộc)

Mỗi dòng: `câu_quốc_ngữ<TAB>câu_hán_nôm`

- Câu Quốc ngữ: các âm tiết cách nhau **một khoảng trắng**.
- Câu Hán/Nôm: các ký tự viết liền nhau (không khoảng trắng) — mỗi ký tự Unicode là một token.
- Hai câu **phải có cùng số đơn vị** sau khi tách token (âm tiết ↔ ký tự theo đúng thứ tự,
  Mục 2.1.4). Cặp nào lệch số lượng sẽ tự động bị loại ở bước tiền xử lý.

```
nam quốc sơn hà nam đế cư	南國山河南帝居
tiệt nhiên định phận tại thiên thư	截然定分在天書
```

### `dict.tsv` — tự điển Hán Việt / Nôm–Quốc ngữ (bắt buộc, dùng mở rộng tập ứng viên)

Mỗi dòng: `ký_tự_hán_nôm<TAB>âm_đọc_quốc_ngữ`

Một ký tự có thể xuất hiện nhiều dòng nếu có nhiều cách đọc (đa âm tự):

```
行	hành
行	hàng
行	hạnh
樂	nhạc
樂	lạc
樂	nhạo
```

### `phrases.tsv` — cụm từ cố định (tuỳ chọn, mặc định tắt trong config)

Mỗi dòng: `cụm_quốc_ngữ_nối_bằng_gạch_dưới<TAB>cụm_hán_nôm`, số đơn vị hai phía phải bằng nhau:

```
nam_quốc	南國
sơn_hà	山河
```

Chỉ dùng khi `phrase_table.enabled: true` trong config — báo cáo ghi nhận phần này cải thiện rất
nhỏ và chưa ổn định (Mục 4.4.3), nên không phải ưu tiên - và hiện repo này không có dạng data này.

## 4. Chạy thử với dữ liệu giả lập 

```bash
bash scripts/run_demo.sh
```

Script này sẽ: sinh dữ liệu giả lập → tiền xử lý → huấn luyện CCT-C và CCT-D vài epoch nhỏ trên
CPU → đánh giá BLEU/token-accuracy → chuyển tự thử một câu. Mục đích chỉ là xác nhận pipeline chạy
đúng logic, **không phản ánh chất lượng thật** (dữ liệu giả lập rất nhỏ và không phải ngữ liệu
Hán Nôm thực).

## 5. Chạy với dữ liệu thật trên GPU/Kaggle

```bash
# 1. Đặt parallel.tsv, dict.tsv (và phrases.tsv nếu dùng) vào data/raw/han/ và data/raw/nom/

# 2. Tiền xử lý — chạy riêng cho từng hệ chữ
python -m src.data_prep --config configs/default.yaml --script han
python -m src.data_prep --config configs/default.yaml --script nom

# 3. Huấn luyện (đổi model_type: cct_c | cct_d | baseline)
python -m src.train --config configs/default.yaml --script han --model_type cct_c
python -m src.train --config configs/default.yaml --script han --model_type cct_d
python -m src.train --config configs/default.yaml --script han --model_type baseline

# 4. Đánh giá trên tập test (điểm BLEU-4 + token accuracy)
python -m src.evaluate --config configs/default.yaml --script han --model_type cct_d \
    --checkpoint checkpoints/han/cct_d/best.pt

# 5. Chuyển tự một câu (Quốc ngữ -> Hán/Nôm)
python -m src.infer --config configs/default.yaml --script han --model_type cct_d \
    --checkpoint checkpoints/han/cct_d/best.pt --text "nam quốc sơn hà"
```

Lặp lại toàn bộ cho `--script nom`. Mỗi hệ chữ (Hán/Nôm) huấn luyện **mô hình riêng**; mỗi mô
hình chỉ có **một chiều** chuyển tự (Quốc ngữ -> Hán/Nôm) — không có chiều ngược lại.

## 6. Kết quả cần báo cáo

Khi so sánh mô hình, tách kết quả theo hệ chữ: {Hán, Nôm} — không gộp chung, vì mức độ khó (số
ứng viên trung bình, tỉ lệ đơn trị) khác nhau giữa hai hệ chữ (Bảng 3.2).

## 7. Giới hạn đã biết của bản cài đặt này

- Chỉ cài đặt một chiều Quốc ngữ -> Hán/Nôm theo đúng yêu cầu của báo cáo này; khóa luận tham
  khảo có cài thêm chiều ngược lại (Hán/Nôm -> Quốc ngữ) và huấn luyện chung hai chiều trong một
  mô hình bằng token chỉ thị chiều — phần đó không nằm trong phạm vi bản cài đặt hiện tại.
- Bảng nhúng đầu ra dùng **một** bảng chung trên toàn bộ từ vựng thống nhất và chỉ giới hạn theo
  tập ứng viên C(xi) khi tính điểm — về mặt toán học tương đương softmax chỉ chạy trên candidate
  của đúng phía đích, nhưng tiết kiệm bộ nhớ hơn so với tách riêng bảng nhúng theo chiều.
- Chưa cài `back-translation` hay tiền huấn luyện — đúng như phạm vi và hướng phát triển nêu ở
  Mục 4.6.2 của báo cáo tham khảo.
- `baseline_transformer.py` là bản viết lại bằng PyTorch thuần (báo cáo dùng fairseq); dùng để có
  con số đối chứng tương đối, không kỳ vọng khớp tuyệt đối với số liệu trong báo cáo.
- Baseline cần đủ số bước tối ưu để vượt qua giai đoạn khởi động (`train_baseline.warmup_steps:
  1000`, đặt theo quy mô ngữ liệu thật hàng trăm nghìn câu của báo cáo). Trên tập dữ liệu giả lập
  rất nhỏ (`scripts/run_demo.sh` chỉ chạy vài epoch để demo), baseline sẽ chưa kịp rời khỏi trạng
  thái gần như đoán ngẫu nhiên và có thể sinh chuỗi rỗng/BLEU thấp — **không phải lỗi cài đặt**, đã
  kiểm chứng bằng cách chạy dài hơn (loss giảm đều đặn). Vì lý do này, `run_demo.sh` không chạy thử
  baseline; chỉ nên đánh giá baseline sau khi huấn luyện đủ epoch trên ngữ liệu thật.
