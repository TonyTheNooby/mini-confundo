from __future__ import annotations  # Trì hoãn đánh giá các chú thích kiểu dữ liệu.
from dataclasses import dataclass  # Nhập công cụ tạo lớp chứa dữ liệu.
from typing import Sequence  # Mô tả chuỗi phần tử có thứ tự, ví dụ list hoặc tuple.
import numpy as np  # Nhập NumPy để xử lý mảng số với tên viết tắt np.

@dataclass(frozen=True)  # Tự tạo hàm khởi tạo và ngăn gán lại các trường; nội dung mảng/list vẫn có thể thay đổi.
class AVFilterResult:  # Lớp chứa kết quả phát hiện các điểm NPAS bất thường.
    keep_mask: np.ndarray  # Mảng boolean: True là giữ đoạn văn, False là loại bỏ.
    suspicious_indices: list[int]  # Danh sách vị trí các đoạn đáng ngờ, đánh số từ 0.
    z_scores: np.ndarray  # Mức chênh lệch so với trung bình, tính theo đơn vị độ lệch chuẩn.
    mean: float  # Giá trị trung bình của các điểm NPAS.
    std: float  # Độ lệch chuẩn của các điểm NPAS.

class AVFilter:  # Bộ lọc đánh dấu các đoạn có điểm NPAS cao bất thường.
    """Bộ lọc anomaly đơn giản trên phân bố NPAS.

    Một passage bị đánh dấu khi NPAS cao hơn mean + z_threshold * std.
    Đây là baseline có thể thay thế bằng rule đúng theo implementation paper khi nhóm
    tái hiện đầy đủ thuật toán/tuning từ mã nguồn tác giả.
    """  # Tài liệu mô tả quy tắc lọc và phạm vi của bản triển khai thử nghiệm.
    def __init__(self, z_threshold: float = 1.5, min_std: float = 1e-8):  # Nhận ngưỡng điểm z và độ lệch chuẩn tối thiểu.
        self.z_threshold = float(z_threshold)  # Lưu ngưỡng dưới dạng số thực; điểm z vượt ngưỡng sẽ bị đánh dấu.
        self.min_std = float(min_std)  # Lưu mức tối thiểu để tránh chia cho độ lệch chuẩn quá nhỏ.

    def detect(self, npas_scores: Sequence[float]) -> AVFilterResult:  # Nhận các điểm NPAS và trả về kết quả phát hiện bất thường.
        x = np.asarray(npas_scores, dtype=float)  # Chuyển các điểm đầu vào thành mảng NumPy kiểu số thực.
        if x.ndim != 1 or len(x) < 2:  # Yêu cầu mảng một chiều có ít nhất hai điểm.
            raise ValueError("Cần ít nhất 2 NPAS score.")  # Dừng xử lý và báo lỗi khi đầu vào không đáp ứng điều kiện.
        mean = float(x.mean())  # Tính trung bình các điểm rồi chuyển thành float của Python.
        std = float(x.std())  # Tính độ lệch chuẩn với ddof=0, tức chia phương sai cho số phần tử.
        if std < self.min_std:  # Kiểm tra trường hợp các điểm gần như bằng nhau.
            z = np.zeros_like(x)  # Gán toàn bộ điểm z bằng 0 để tránh phép chia không ổn định.
        else:  # Xử lý khi độ lệch chuẩn đủ lớn để tính điểm z.
            z = (x - mean) / std  # Chuẩn hóa mức chênh lệch của từng điểm so với trung bình.
        suspicious = np.flatnonzero(z > self.z_threshold).tolist()  # Lấy các chỉ số có điểm z vượt ngưỡng và chuyển thành list.
        keep = np.ones(len(x), dtype=bool)  # Tạo mặt nạ có cùng số phần tử, ban đầu giữ tất cả.
        keep[suspicious] = False  # Đánh dấu loại bỏ các phần tử đáng ngờ.
        return AVFilterResult(keep, suspicious, z, mean, std)  # Đóng gói kết quả theo thứ tự các trường của AVFilterResult.

    def filter(self, passages: Sequence[str], npas_scores: Sequence[float]):  # Nhận các đoạn văn cùng điểm NPAS tương ứng để lọc.
        if len(passages) != len(npas_scores):  # Kiểm tra mỗi đoạn văn có đúng một điểm tương ứng.
            raise ValueError("Số passage và NPAS score phải bằng nhau.")  # Báo lỗi nếu số đoạn văn và số điểm không khớp.
        result = self.detect(npas_scores)  # Phát hiện các điểm bất thường và tạo mặt nạ giữ lại.
        cleaned = [p for p, keep in zip(passages, result.keep_mask) if keep]  # Ghép đoạn văn với mặt nạ và lấy các đoạn có True.
        removed = [p for p, keep in zip(passages, result.keep_mask) if not keep]  # Lấy các đoạn có False trong mặt nạ.
        return cleaned, removed, result  # Trả về các đoạn giữ lại, các đoạn bị loại và chi tiết phát hiện.
