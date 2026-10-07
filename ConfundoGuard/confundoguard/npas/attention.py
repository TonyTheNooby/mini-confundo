from __future__ import annotations  # Trì hoãn đánh giá các chú thích kiểu dữ liệu.
from typing import Iterable  # Mô tả đầu vào có thể duyệt, ví dụ list hoặc generator.
import numpy as np  # Nhập NumPy để xử lý và tổng hợp mảng số.


def aggregate_response_attention(attention_steps: Iterable[np.ndarray]) -> np.ndarray:  # Tổng hợp các vector attention của các bước sinh câu trả lời.
    """Gộp attention qua các token response.

    Mỗi phần tử là vector attention lên input tokens ở một bước sinh.
    Trả về mean attention vector. Adapter model cụ thể nên chuyển tensor -> numpy trước.
    """  # Mô tả dạng đầu vào, đầu ra và trách nhiệm chuyển đổi tensor của adapter.
    steps = [np.asarray(x, dtype=float).reshape(-1) for x in attention_steps]  # Chuyển từng bước thành mảng số thực một chiều rồi gom vào list.
    if not steps:  # Kiểm tra có ít nhất một bước attention.
        raise ValueError("Không có attention step nào.")  # Báo lỗi nếu không có dữ liệu để tổng hợp.
    n = len(steps[0])  # Lấy độ dài vector đầu tiên làm độ dài chuẩn.
    if any(len(x) != n for x in steps):  # Kiểm tra tất cả vector có cùng độ dài.
        raise ValueError("Các attention vector phải cùng độ dài.")  # Báo lỗi vì các vector khác độ dài không thể xếp thành ma trận đều.
    return np.mean(np.stack(steps, axis=0), axis=0)  # Xếp mỗi bước thành một hàng và lấy trung bình theo từng vị trí token.
