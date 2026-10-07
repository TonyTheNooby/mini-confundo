# ConfundoGuard

Thư viện Python mẫu phục vụ **ĐATN về phòng thủ Confundo / RAG Poisoning có Stealthiness Optimization (tối ưu khả năng ẩn mình)**.

## 1. Mục tiêu

Project tập trung vào 2 thành phần phòng thủ:

1. **NPAS – Normalized Passage Attention Score (Điểm chú ý chuẩn hóa của từng đoạn tài liệu):** tổng hợp tín hiệu Attention (mức chú ý/ảnh hưởng) ở cấp passage và chuẩn hóa thành phân phối.
2. **AV Filter – Attention-Variance Filter (Bộ lọc phương sai Attention):** sử dụng sự bất thường trong phân bố NPAS để đánh dấu/lọc passage đáng ngờ.

Luồng nghiên cứu:

```text
Confundo / RAG Poisoning
        ↓
Stealthiness Optimization
        ↓
Stealthy Poisoned Passages
        ↓
Retriever → Retrieved Passages
        ↓
NPAS Analysis
        ↓
AV Filter
        ↓
Cleaned Context → LLM → Final Response
```

> **Lưu ý học thuật quan trọng:** Project này là **khung tái hiện/thực nghiệm**. `AVFilter` hiện cung cấp một baseline anomaly rule đơn giản (`mean + z_threshold × std`) để project chạy độc lập. Không nên tuyên bố rule này là bản sao chính xác 100% thuật toán của paper nếu chưa đối chiếu implementation chính thức, hyperparameter và model setup của tác giả. Phần đóng góp mới của ĐATN cần được tách rõ khỏi phần tái hiện paper.

## 2. Cài đặt trên Windows + VS Code

### Bước 1 – Cài Python
Cài Python 3.10 trở lên. Khi cài trên Windows, bật **Add Python to PATH (thêm Python vào biến môi trường PATH)**.

Kiểm tra trong CMD/PowerShell:

```powershell
python --version
```

### Bước 2 – Mở project trong VS Code
Giải nén `ConfundoGuard.zip` → mở VS Code → **File → Open Folder** → chọn thư mục `ConfundoGuard`.

Mở Terminal trong VS Code bằng **Terminal → New Terminal**.

### Bước 3 – Tạo môi trường ảo

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script, có thể dùng CMD:

```cmd
.venv\Scripts\activate.bat
```

### Bước 4 – Cài thư viện

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` cài ConfundoGuard ở chế độ editable (chỉnh code là dùng ngay).

## 3. Chạy demo

```powershell
python examples/rag_defense_demo.py
```

Demo tạo 5 passage, trong đó passage số 3 có influence cao. Pipeline chuẩn hóa thành NPAS rồi AV Filter đánh dấu passage bất thường.

## 4. Chạy kiểm thử

```powershell
pytest -q
```

Nếu hiện `4 passed` nghĩa là các thành phần lõi hoạt động.

## 5. Giải thích từng file

```text
ConfundoGuard/
├── confundoguard/
│   ├── npas/
│   │   ├── attention.py
│   │   └── scorer.py
│   ├── filters/
│   │   └── av_filter.py
│   ├── defense/
│   │   └── pipeline.py
│   ├── metrics/
│   │   └── evaluator.py
│   └── __init__.py
├── examples/
│   └── rag_defense_demo.py
├── tests/
│   └── test_core.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

### `npas/attention.py`
Nhận Attention theo từng bước sinh response và gộp thành một vector Attention đại diện trên input tokens.

### `npas/scorer.py`
- `compute_npas(...)`: cộng Attention theo vùng token của từng passage rồi chuẩn hóa.
- `normalize_passage_influence(...)`: dùng khi đã có influence score ở cấp passage.

### `filters/av_filter.py`
Chứa `AVFilter`. Bản baseline tính mean (trung bình), standard deviation (độ lệch chuẩn) và z-score. Passage có NPAS cao bất thường được đánh dấu.

### `defense/pipeline.py`
Ghép NPAS và AV Filter thành `ConfundoGuard`. Đây là API chính mà code RAG có thể gọi.

### `metrics/evaluator.py`
Tính các chỉ số phục vụ ĐATN:
- **Detection Rate / Recall (tỷ lệ phát hiện)**
- **FPR – False Positive Rate (tỷ lệ dương tính giả)**
- **Precision (độ chính xác phát hiện)**
- **F1-score**
- **ASR – Attack Success Rate (tỷ lệ tấn công thành công)**

### `examples/rag_defense_demo.py`
Ví dụ nhỏ để kiểm tra pipeline mà chưa cần tải LLM lớn.

### `tests/test_core.py`
Unit test (kiểm thử đơn vị) cho NPAS, AV Filter và metrics.

### `pyproject.toml`
Khai báo package Python để có thể `pip install -e .`.

## 6. Cách dùng trong code

```python
from confundoguard import ConfundoGuard

passages = ["passage A", "passage B", "passage C"]
influence = [0.10, 0.75, 0.15]

guard = ConfundoGuard(z_threshold=1.0)
result = guard.defend_from_passage_influence(passages, influence)

print(result.npas_scores)
print(result.removed_passages)
print(result.cleaned_passages)
```

Trong hệ thống thật, `influence` phải được lấy từ Attention/model signal thay vì tự nhập như demo.

## 7. Thực nghiệm đúng với đề tài Stealthiness Optimization

Không chỉ kiểm tra poison thông thường. Nên tạo ba nhóm:

```text
A. Clean RAG
B. Confundo không Stealthiness Optimization
C. Confundo + Stealthiness Optimization
```

Chạy B và C qua cùng ConfundoGuard rồi so sánh:

| Chỉ số | Ý nghĩa |
|---|---|
| Detection Rate | Defense bắt được bao nhiêu poison |
| FPR | Bao nhiêu passage sạch bị nhận nhầm |
| ASR trước defense | Attack mạnh đến đâu trước phòng thủ |
| ASR sau defense | Attack còn thành công bao nhiêu sau lọc |
| ASR reduction | Defense làm giảm ASR bao nhiêu |

Điểm nghiên cứu quan trọng là **C**: nếu Stealthiness Optimization làm giảm dấu vết Attention, NPAS/AV Filter có thể suy yếu. Đây là nơi nhóm có thể nghiên cứu thêm tín hiệu khác hoặc cơ chế đa tầng.

## 8. Giới hạn hiện tại

- Chưa kèm model adapter cụ thể cho Llama/Mistral/etc. vì cách lấy Attention phụ thuộc kiến trúc/model và phiên bản `transformers`.
- Demo không tự tạo poison và không triển khai Confundo attack; mục tiêu package này là **defense/evaluation**.
- Threshold của AV Filter cần tune (hiệu chỉnh) trên validation data (dữ liệu xác thực), không nên chọn một con số rồi kết luận chung.
- Attention là proxy (tín hiệu đại diện), không nên mô tả là bằng chứng nhân quả tuyệt đối.

## 9. Hướng mở rộng cho ĐATN

Bước tiếp theo hợp lý là thêm `adapters/` để lấy Attention trực tiếp từ một LLM mã nguồn mở, loader cho dataset/Confundo output, script benchmark so sánh **không Stealthiness Optimization vs có Stealthiness Optimization**, và xuất CSV/biểu đồ ASR–Detection–FPR.
