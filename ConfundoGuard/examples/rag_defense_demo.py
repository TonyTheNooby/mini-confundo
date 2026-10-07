from confundoguard import ConfundoGuard

passages = [
    "Passage 1: thông tin sạch A.",
    "Passage 2: thông tin sạch B.",
    "Passage 3: nội dung bị đầu độc nhưng được viết tự nhiên.",
    "Passage 4: thông tin sạch C.",
    "Passage 5: thông tin sạch D.",
]
# Demo: influence/attention aggregate đã được adapter model tính trước.
influence = [0.10, 0.11, 0.55, 0.12, 0.12]

guard = ConfundoGuard(z_threshold=1.5)
out = guard.defend_from_passage_influence(passages, influence)

print("NPAS:", [round(x, 4) for x in out.npas_scores])
print("Passage bị nghi ngờ:", out.filter_result.suspicious_indices)
print("Đã loại:", out.removed_passages)
print("Context còn lại:", out.cleaned_passages)
