# Datathon 2026 — Sales Forecasting Model

> Mô hình dự báo doanh thu (Revenue) và giá vốn hàng bán (COGS) hàng ngày cho công ty thương mại điện tử thời trang Việt Nam, giai đoạn 01/01/2023 – 01/07/2024.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Chuẩn bị dữ liệu](#-chuẩn-bị-dữ-liệu)
- [Cách chạy](#-cách-chạy)
- [Phương pháp tiếp cận](#-phương-pháp-tiếp-cận)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Cấu trúc đầu ra](#-cấu-trúc-đầu-ra)
- [Khắc phục sự cố](#-khắc-phục-sự-cố)
- [Tác giả](#-tác-giả)

---

## 🎯 Giới thiệu

Dự án này xây dựng pipeline machine learning để dự báo doanh thu và giá vốn hàng bán hàng ngày dựa trên dữ liệu lịch sử 2012-2022 của một doanh nghiệp thương mại điện tử thời trang.

**Bài toán:**
- **Input**: Dữ liệu lịch sử doanh thu, đơn hàng, khuyến mãi từ 04/07/2012 đến 31/12/2022
- **Output**: Dự báo Revenue và COGS hàng ngày cho 548 ngày (01/01/2023 – 01/07/2024)
- **Đánh giá**: MAE, RMSE, R²

**Điểm nổi bật:**
- ✅ Phát hiện structural break 2018→2019 và xử lý bằng cách chỉ train trên dữ liệu ổn định 2019-2022
- ✅ Ensemble HistGradientBoosting + Ridge với 51 features tối ưu
- ✅ Recursive day-by-day prediction để bắt pattern dài hạn
- ✅ Validation R² = 0.93 trên dữ liệu out-of-sample 2022

---

## 📁 Cấu trúc thư mục

```
datathon-2026/
├── README.md                    ← File này
├── requirements.txt             ← Python dependencies
├── main.py                      ← Code chính
├── data/                        ← Thư mục dữ liệu (cần tự tạo)
│   ├── sales.csv
│   ├── orders.csv
│   ├── promotions.csv
│   └── sample_submission.csv
└── submission.csv               ← File output (được tạo sau khi chạy)
```

---

## 🛠 Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **RAM**: tối thiểu 4GB (khuyến nghị 8GB)
- **Dung lượng ổ cứng**: ~200MB cho dữ liệu + thư viện
- **Hệ điều hành**: Windows / macOS / Linux

---

## 📦 Cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/<username>/datathon-2026.git
cd datathon-2026
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

Tạo file `requirements.txt` với nội dung:

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

Sau đó chạy:

```bash
pip install -r requirements.txt
```

Hoặc cài trực tiếp:

```bash
pip install numpy pandas scikit-learn
```

---

## 📊 Chuẩn bị dữ liệu

### Bước 1: Tạo thư mục `data/`

```bash
mkdir data
```

### Bước 2: Đặt 4 file CSV vào thư mục `data/`

Cần có các file sau (lấy từ bộ dataset Datathon 2026):

| File | Mô tả | Cột chính |
|------|-------|-----------|
| `sales.csv` | Doanh thu hàng ngày 2012-2022 | Date, Revenue, COGS |
| `orders.csv` | Đơn hàng chi tiết | order_id, order_date, customer_id, ... |
| `promotions.csv` | Lịch khuyến mãi | promo_id, start_date, end_date, promo_type, ... |
| `sample_submission.csv` | Format file nộp | Date, Revenue, COGS |

### Bước 3: ⚠️ QUAN TRỌNG — Cấu hình đường dẫn dữ liệu

Mở file `main.py` và **sửa biến `DATA_DIR`** ở đầu file (dòng ~10) cho phù hợp với môi trường của bạn:

```python
# Trên máy tính cá nhân (data nằm cùng thư mục với main.py)
DATA_DIR = "./data"

# Trên Kaggle Notebook
DATA_DIR = "/kaggle/input/<your-dataset-name>"

# Trên Google Colab (sau khi mount Drive)
DATA_DIR = "/content/drive/MyDrive/datathon-2026/data"

# Trên Linux/Docker với volume mount
DATA_DIR = "/data"
```

**Lưu ý:** Mặc định trong code là `DATA_DIR = "/data"` (Linux absolute path). Nếu bạn chạy trên Windows/macOS, **bắt buộc** phải đổi thành `"./data"` (đường dẫn tương đối) hoặc đường dẫn tuyệt đối tới thư mục data của bạn.

**Cách kiểm tra nhanh:** Mở terminal trong thư mục project và chạy:

```bash
ls data/
```

Nếu thấy 4 file `.csv` thì đường dẫn `./data` là đúng.

---

## 🚀 Cách chạy

### ⚠️ Trước khi chạy — Checklist

- [ ] Đã cài đầy đủ thư viện trong `requirements.txt`
- [ ] Đã đặt 4 file CSV vào thư mục `data/`
- [ ] **Đã sửa `DATA_DIR` trong `main.py`** cho phù hợp với môi trường (xem [Bước 3](#bước-3--quan-trọng--cấu-hình-đường-dẫn-dữ-liệu) phía trên)

### Chạy đơn giản

```bash
python main.py
```

### Output mong đợi

Sau khi chạy thành công, terminal sẽ in ra:

```
1. Loading data...
   Train: 2012-07-04 → 2022-12-31
   Test : 2023-01-01 → 2024-07-01 (548 days)
2. Aggregating signals...
3. Building promo features...
4. Building master dataframe...
5. Computing lag features...
6. Training on 2019-2021 (stable period only)...
   Train: 1096 rows  |  Val: 365 rows

── Validation 2022 ──
   [Revenue] MAE=     325,442  RMSE=     443,696  R²=0.9297
   [COGS   ] MAE=     327,730  RMSE=     453,653  R²=0.9033

7. Retraining on full 2019-2022...
   Trained on 1461 rows (2019-2022)

8. Recursive day-by-day prediction...
   ✅ Revenue range: 1,836,655 – 3,848,569
      COGS    range: 1,771,312 – 3,712,208

9. Building submission...
   ✅ Saved 548 rows → submission.csv
```

**Thời gian chạy:** ~2-5 phút trên máy tính cá nhân, ~1 phút trên Kaggle.

---

## 🧠 Phương pháp tiếp cận

### Phát hiện cốt lõi

Khi phân tích doanh thu 2012-2022, chúng tôi phát hiện **structural break** vào 2018-2019: doanh thu trung bình giảm từ 4-6 triệu USD/ngày xuống còn 2.8-3.2 triệu USD/ngày. Train trên toàn bộ 10 năm sẽ khiến mô hình bị bias và dự đoán quá lớn cho 2023-2024.

**Giải pháp:** Chỉ sử dụng dữ liệu **2019-2022** (giai đoạn phân phối ổn định).

### Pipeline mô hình

```
┌─────────────────────────────────────────────────────────┐
│  1. LOAD DATA                                           │
│     sales.csv + orders.csv + promotions.csv             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  2. FEATURE ENGINEERING (51 features)                   │
│     ├─ Calendar: dayofweek, month, weekofyear, ...      │
│     ├─ Holidays: is_vn_holiday (Tết, 30/4, 1/5, 2/9)    │
│     ├─ Fourier: 4 weekly + 8 yearly harmonics           │
│     ├─ Year-ago lags: 363-367, 728-731 days             │
│     ├─ Short lags: 7, 14, 21, 28 days                   │
│     ├─ Rolling: mean/std on 7/14/30/90/180/365 windows  │
│     └─ YoY ratios + COGS/Revenue ratio                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  3. TRAIN / VALIDATE                                    │
│     Train: 2019-01-01 → 2021-12-31 (1,096 days)         │
│     Val:   2022-01-01 → 2022-12-31 (365 days)           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  4. ENSEMBLE                                            │
│     0.92 × HistGradientBoosting + 0.08 × Ridge          │
│     Target: log1p(Revenue)                              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  5. RETRAIN ON FULL DATA (2019-2022)                    │
│     Tận dụng 365 ngày 2022 sau khi xác nhận hyperparams │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  6. RECURSIVE PREDICTION                                │
│     Predict day-by-day, update rev_arr cho ngày sau     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  7. COGS = Revenue × cogs_ratio_roll90                  │
└─────────────────────────────────────────────────────────┘
```

### Hyperparameters chính

```python
HistGradientBoostingRegressor(
    max_iter=2000,
    learning_rate=0.01,
    max_depth=6,
    min_samples_leaf=10,
    l2_regularization=0.3,
    max_bins=255,
    random_state=42
)

Ridge(alpha=50)
```

---

## 📈 Kết quả thực nghiệm

### Validation 2022 (out-of-sample)

| Metric | Revenue | COGS |
|--------|---------|------|
| MAE | 325,442 | 327,730 |
| RMSE | 443,696 | 453,653 |
| **R²** | **0.9297** | **0.9033** |

### So sánh với baselines

| Phương pháp | MAE Revenue | R² Revenue |
|-------------|-------------|------------|
| Naive — pure 2021 | 837,704 | 0.518 |
| Naive — avg 2019-2021 | 677,088 | 0.689 |
| Naive — weighted | 696,680 | 0.669 |
| **Phương pháp đề xuất** | **325,442** | **0.9297** |

→ Mô hình giảm MAE **52%** so với naive average tốt nhất.

### Top features quan trọng

1. `rev_lag365` — Doanh thu cùng ngày năm trước
2. `rev_lag364`, `rev_lag366` — Doanh thu xung quanh cùng kỳ
3. `rev_roll90` — Trung bình 90 ngày qua
4. `ord_lag365` — Số đơn hàng năm trước
5. `cogs_ratio_roll90` — Tỷ lệ COGS/Revenue gần nhất
6. `dayofyear`, `month` — Mùa vụ
7. `sy1`, `sy2` — Fourier năm bậc thấp
8. `is_promo` — Cờ khuyến mãi
9. `rev_yoy` — Tốc độ tăng trưởng YoY

---

## 📤 Cấu trúc đầu ra

File `submission.csv` được tạo với format:

```csv
Date,Revenue,COGS
2023-01-01,2039154.48,2197492.94
2023-01-02,1137772.75,1187250.12
2023-01-03,1115453.39,1061072.85
...
2024-07-01,3245893.21,2843105.67
```

- **548 dòng**, đúng thứ tự với `sample_submission.csv`
- Đơn vị: VNĐ
- Encoding: UTF-8

---

## 🔧 Khắc phục sự cố

### Lỗi: `FileNotFoundError: /data/sales.csv`

**Nguyên nhân:** Đường dẫn dữ liệu sai.

**Cách sửa:** Mở `main.py`, sửa `DATA_DIR` thành đường dẫn thực tế:

```python
DATA_DIR = "./data"   # nếu data nằm cùng thư mục với main.py
```

### Lỗi: `ModuleNotFoundError: No module named 'sklearn'`

**Cách sửa:**
```bash
pip install scikit-learn
```

### Lỗi: Memory Error

**Cách sửa:** Giảm `max_iter` từ 2000 xuống 1000:

```python
HGB_PARAMS = dict(
    max_iter=1000,   # giảm từ 2000
    ...
)
```

### Validation MAE quá cao (>500K)

**Có thể do:**
- File dữ liệu bị thiếu dòng → kiểm tra `sales.csv` có đủ 3,834 dòng
- Đường dẫn `DATA_DIR` sai → in ra `df.shape` để kiểm tra
- Chưa cài đúng version scikit-learn → cập nhật: `pip install --upgrade scikit-learn`

---

## 📝 Tính tái lập (Reproducibility)

Toàn bộ pipeline đều có random seed cố định:

```python
SEED = 42
np.random.seed(SEED)
```

Mỗi mô hình (HistGradientBoosting, Ridge) đều có `random_state=42`. Chạy nhiều lần sẽ ra kết quả **giống hệt nhau**.

---

## 👤 Tác giả

**Đỉnh Nhật Minh Phạm**

- Datathon 2026 — Round 1
- Được tổ chức bởi VinTelligence — VinUni DS&AI Club

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

- VinTelligence — VinUni DS&AI Club tổ chức cuộc thi
- scikit-learn team cho thư viện ML
- Cộng đồng Kaggle cho các kỹ thuật time-series forecasting

---

<p align="center">
  Nếu repository này hữu ích, hãy ⭐ star để ủng hộ tác giả!
</p>
