from __future__ import annotations  # Trì hoãn đánh giá các chú thích kiểu dữ liệu.
from dataclasses import dataclass  # Nhập công cụ tạo lớp chứa dữ liệu.
from typing import Sequence  # Mô tả chuỗi phần tử có thứ tự, ví dụ list hoặc tuple.
import numpy as np  # Nhập NumPy để xử lý mảng và tính tổng điểm.

@dataclass(frozen=True)  # Ngăn gán lại các trường; nội dung mảng bên trong vẫn có thể thay đổi.
class NPASResult:  # Chứa điểm chuẩn hóa theo đoạn văn và điểm theo token.
    scores: np.ndarray  # Mảng điểm NPAS của từng đoạn văn.
    token_scores: np.ndarray  # Vector attention theo token sau khi đưa các giá trị âm về 0.


def compute_npas(attention: np.ndarray, passage_token_spans: Sequence[tuple[int, int]], *, eps: float = 1e-12) -> NPASResult:  # Tính NPAS theo phạm vi token; eps chỉ được truyền bằng tên tham số.
    """Tính NPAS từ attention đã trích xuất.

    attention: vector 1D [seq_len], thể hiện mức attention của response lên từng token input.
    passage_token_spans: danh sách (start, end), end-exclusive cho từng passage.

    Đây là implementation thực nghiệm tối giản để đóng gói pipeline. Việc trích attention
    phụ thuộc model/transformers và được tách khỏi hàm này.
    """  # Mô tả đầu vào: mỗi phạm vi bao gồm start và không bao gồm end.
    a = np.asarray(attention, dtype=float).reshape(-1)  # Chuyển attention thành vector số thực một chiều.
    a = np.clip(a, 0.0, None)  # Đưa giá trị âm về 0, không đặt giới hạn trên.
    passage_mass = []  # Tạo danh sách chứa tổng attention của từng đoạn văn.
    for start, end in passage_token_spans:  # Duyệt vị trí bắt đầu và kết thúc của từng đoạn.
        if start < 0 or end <= start or end > len(a):  # Kiểm tra phạm vi không âm, không rỗng và không vượt độ dài vector.
            raise ValueError(f"Span không hợp lệ: {(start, end)} với seq_len={len(a)}")  # Báo lỗi kèm phạm vi sai và độ dài vector.
        passage_mass.append(float(a[start:end].sum()))  # Cộng attention từ start đến trước end và lưu tổng của đoạn.
    mass = np.asarray(passage_mass, dtype=float)  # Chuyển danh sách tổng attention thành mảng số thực.
    total = mass.sum()  # Tính tổng attention trên tất cả các phạm vi được cung cấp.
    scores = mass / (total + eps)  # Chuẩn hóa theo tổng cộng epsilon; tổng điểm gần 1 khi total lớn hơn eps nhiều lần.
    return NPASResult(scores=scores, token_scores=a)  # Trả về điểm theo đoạn cùng vector attention đã xử lý.


def normalize_passage_influence(raw_scores: Sequence[float], *, eps: float = 1e-12) -> np.ndarray:  # Chuẩn hóa điểm ảnh hưởng có sẵn; eps chỉ được truyền bằng tên tham số.
    """Chuẩn hóa các influence score passage thành phân phối NPAS tổng bằng 1."""  # Mục tiêu là tổng gần 1; nếu mọi điểm bằng 0 thì kết quả cũng bằng 0.
    x = np.asarray(raw_scores, dtype=float)  # Chuyển điểm đầu vào thành mảng số thực.
    x = np.clip(x, 0.0, None)  # Đưa các điểm âm về 0 và giữ nguyên các điểm không âm.
    return x / (x.sum() + eps)  # Chia mỗi điểm cho tổng cộng epsilon để tránh chia cho 0 với eps mặc định.
