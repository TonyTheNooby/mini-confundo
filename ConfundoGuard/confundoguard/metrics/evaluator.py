from __future__ import annotations  # Trì hoãn đánh giá các chú thích kiểu dữ liệu.
from dataclasses import dataclass  # Nhập công cụ tạo lớp chứa dữ liệu.

@dataclass(frozen=True)  # Tự tạo hàm khởi tạo và ngăn gán lại các trường kết quả.
class DetectionMetrics:  # Chứa các chỉ số đánh giá bộ phát hiện; nhãn 1 là dương tính, nhãn 0 là âm tính.
    detection_rate: float  # Tỷ lệ phát hiện đúng các mẫu dương tính; ở đây bằng recall.
    false_positive_rate: float  # Tỷ lệ mẫu âm tính bị nhận nhầm là dương tính.
    precision: float  # Tỷ lệ dự đoán dương tính thực sự đúng.
    recall: float  # Tỷ lệ mẫu dương tính thực tế được phát hiện.
    f1: float  # Trung bình điều hòa của precision và recall.


def detection_metrics(y_true: list[int], y_pred: list[int]) -> DetectionMetrics:  # Tính chỉ số từ nhãn thực tế và nhãn dự đoán tương ứng.
    if len(y_true) != len(y_pred) or not y_true:  # Kiểm tra hai danh sách có cùng độ dài và không rỗng.
        raise ValueError("y_true/y_pred phải cùng độ dài và không rỗng.")  # Báo lỗi khi dữ liệu không đáp ứng điều kiện.
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))  # Đếm true positive: thực tế dương tính và dự đoán dương tính.
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred))  # Đếm true negative: thực tế âm tính và dự đoán âm tính.
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))  # Đếm false positive: thực tế âm tính nhưng dự đoán dương tính.
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))  # Đếm false negative: thực tế dương tính nhưng dự đoán âm tính.
    recall = tp / (tp + fn) if tp + fn else 0.0  # Chia số phát hiện đúng cho tổng mẫu dương tính; trả 0 nếu mẫu số bằng 0.
    fpr = fp / (fp + tn) if fp + tn else 0.0  # Chia số báo nhầm cho tổng mẫu âm tính; tránh chia cho 0.
    precision = tp / (tp + fp) if tp + fp else 0.0  # Chia số dự đoán dương tính đúng cho tổng dự đoán dương tính.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0  # Tính F1; trả 0 khi tổng precision và recall bằng 0.
    return DetectionMetrics(recall, fpr, precision, recall, f1)  # Đóng gói kết quả; detection_rate dùng cùng giá trị với recall.


def attack_success_rate(success_flags: list[bool]) -> float:  # Tính tỷ lệ tấn công thành công từ danh sách cờ True/False.
    if not success_flags:  # Kiểm tra danh sách có rỗng hay không.
        raise ValueError("Danh sách success_flags rỗng.")  # Báo lỗi vì không thể tính tỷ lệ khi không có mẫu.
    return sum(bool(x) for x in success_flags) / len(success_flags)  # Đếm các cờ True rồi chia cho tổng số lần tấn công.
