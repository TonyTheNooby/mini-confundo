from __future__ import annotations  # Trì hoãn đánh giá các chú thích kiểu dữ liệu.
from dataclasses import dataclass  # Tự tạo hàm khởi tạo và các phương thức cho lớp dữ liệu.
from typing import Sequence  # Mô tả chuỗi phần tử có thứ tự, ví dụ list hoặc tuple.
import numpy as np  # Nhập NumPy để xử lý mảng số với tên viết tắt np.
from ..npas.scorer import normalize_passage_influence  # Nhập hàm chuẩn hóa điểm ảnh hưởng của các đoạn văn.
from ..filters.av_filter import AVFilter, AVFilterResult  # Nhập bộ lọc và kiểu kết quả lọc.

@dataclass  # Tự tạo __init__ từ các trường dữ liệu bên dưới.
class DefenseOutput:  # Lớp chứa kết quả của quá trình phòng thủ.
    npas_scores: np.ndarray  # Mảng điểm ảnh hưởng đã được chuẩn hóa.
    cleaned_passages: list[str]  # Các đoạn văn được bộ lọc giữ lại.
    removed_passages: list[str]  # Các đoạn văn bị bộ lọc loại bỏ.
    filter_result: AVFilterResult  # Chi tiết lọc: mặt nạ giữ lại, chỉ số đáng ngờ và thống kê.

class ConfundoGuard:  # Điều phối bước chuẩn hóa điểm và lọc các đoạn văn.
    """Pipeline NPAS + AV Filter cho thí nghiệm phòng thủ RAG poisoning.

    Với ngưỡng mặc định 1.5, nhóm chỉ có 2–3 passage không thể có
    z-score vượt ngưỡng; cần chọn ngưỡng phù hợp với dữ liệu thực nghiệm.
    """
    def __init__(self, z_threshold: float = 1.5):  # Nhận ngưỡng điểm z; mặc định là 1.5.
        z_threshold = float(z_threshold)
        if not np.isfinite(z_threshold) or z_threshold < 0:
            raise ValueError("z_threshold phải là số hữu hạn không âm.")
        self.av_filter = AVFilter(z_threshold=z_threshold)  # Tạo bộ lọc và lưu vào đối tượng hiện tại.

    def defend_from_passage_influence(self, passages: Sequence[str], influence_scores: Sequence[float]) -> DefenseOutput:  # Nhận các đoạn văn và điểm tương ứng, trả về kết quả phòng thủ.
        scores = np.asarray(influence_scores, dtype=float)
        if scores.ndim != 1:
            raise ValueError("influence_scores phải là mảng một chiều.")
        if len(passages) != len(scores):
            raise ValueError("Số passage và influence score phải bằng nhau.")
        if len(scores) < 2:
            raise ValueError("Cần ít nhất 2 passage và influence score.")
        if not np.isfinite(scores).all():
            raise ValueError("influence_scores không được chứa NaN hoặc Inf.")

        # Chỉ thu nhỏ khi cần để tổng các điểm hữu hạn không bị tràn số.
        positive_scores = np.clip(scores, 0.0, None)
        largest_score = float(positive_scores.max())
        if largest_score > np.finfo(float).max / len(scores):
            scores = positive_scores / largest_score
        npas = normalize_passage_influence(scores)
        cleaned, removed, result = self.av_filter.filter(passages, npas)  # Lọc theo điểm NPAS, lấy các đoạn giữ lại, bị loại và chi tiết lọc.
        return DefenseOutput(npas, cleaned, removed, result)  # Đóng gói bốn giá trị theo thứ tự các trường của DefenseOutput.
