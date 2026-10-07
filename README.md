
# [LAYER 1] RETRIEVAL DEFENSE
# ===========================================================================

# ===========================================================================
# [LAYER 2] GENERATION DEFENSE
# ===========================================================================

# ===========================================================================
# [LAYER 3] STEALTHINESS DEFENSE
# ===========================================================================

# ===========================================================================
# [LAYER 4] PIPELINE / FRAGMENTATION DEFENSE
# ===========================================================================

# ===========================================================================
# MASTER FRAMEWORK — COMBINES ALL FOUR LAYERS
# =========================================================================


Bốn lớp đang hoạt động theo logic:

                         Retrieved chunks
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      [1] Retrieval      [2] Generation    [3] Stealthiness
          Defense            Defense            Defense
             │                 │                 │
             └────────────┬────┴───────┬─────────┘
                          │            │
                          ▼            ▼
                    [4] Pipeline Defense
                          │
                          ▼
                 Four risk scores/chunk
                          │
                          ▼
                     Overall Risk
                          │
                  ┌───────┴────────┐
                  │                │
               Safe           Suspicious
                  │                │
                  │             Remove
                  └───────┬────────┘
                          ▼
                   Safe context
                          │
                          ▼
                     Final LLM



Cụ thể, Retrieval Defense xem chunk nào có embedding similarity với query cao bất thường. Generation Defense dùng leave-one-out: bỏ từng chunk và xem câu trả lời thay đổi bao nhiêu. Stealthiness Defense không chỉ nhìn PPL mà dùng semantic outlier, provenance nếu có và một số dấu hiệu meta/instruction. Pipeline Defense re-chunk ở nhiều kích thước để xem nội dung đáng ngờ có tiếp tục giữ relevance sau fragmentation hay không.
Cuối cùng ConfundoDefenseFramework tổng hợp:


overall_risk = (
    retrieval_weight * retrieval_risk
    + generation_weight * generation_risk
    + stealth_weight * stealth_risk
    + pipeline_weight * pipeline_risk
)



Mặc định mình đang để:

Retrieval       25%
Generation      40%
Stealthiness    15%
Pipeline        20%



Đây chỉ là hyperparameter khởi đầu, không nên ghi trong báo cáo là các trọng số tối ưu. Sau này bạn phải tune bằng experiment.
Một điểm mình chủ động sửa so với Generation Defense trước: framework không cần biết target của attacker để phát hiện. target_answer chỉ là optional để làm diagnostic khi nghiên cứu. Đây quan trọng, vì trong thực tế defender sẽ không biết attacker muốn model trả lời "Voldemort" hay "Nova City".
Phần “vỏ ngoài” nên làm gì?
Mình khuyên kiến trúc cuối cùng nên là Python package làm lõi + Streamlit app làm demo, chứ không nên chỉ dừng ở một file .py.
- File Python đơn: tốt ở giai đoạn hiện tại để nghiên cứu và debug. File mình vừa làm chính là prototype.
- Python package: lựa chọn tốt nhất cho sản phẩm đồ án. Có thể pip install rồi import vào bất kỳ RAG pipeline nào.
- Streamlit app: rất phù hợp làm giao diện demo cho thầy. Nhập query, context, bấm Analyze rồi hiện 4 risk score và chunk nào bị loại.
- CLI: nên có thêm, ví dụ ragshield analyze data.json, tiện chạy benchmark.
- FastAPI/REST API: làm sau nếu muốn tích hợp vào hệ thống RAG khác.
- Browser extension: mình không khuyên. Extension thường không nhìn được retrieved chunks và generation pipeline ở backend, trong khi defense của bạn cần can thiệp đúng vào context trước LLM.
Kiến trúc mình khuyên cho đồ án cuối cùng là:

ragshield/
│
├── ragshield/                  ← Python package chính
│   ├── retrieval.py
│   ├── generation.py
│   ├── stealth.py
│   ├── pipeline.py
│   ├── scoring.py
│   └── guard.py
│
├── app.py                      ← Streamlit demo
├── cli.py                      ← command line
│
├── experiments/
│   ├── eval_clean.py
│   ├── eval_confundo.py
│   └── ablation.py
│
├── tests/
├── README.md
└── pyproject.toml




Người dùng cuối có thể dùng cực đơn giản:

from ragshield import RAGShield

guard = RAGShield()

result = guard.analyze(
    query=query,
    chunks=retrieved_chunks,
    answer_fn=rag_generate
)

rồi:

print(result.suspicious_indices)
print(result.defended_answer)







Còn khi demo cho thầy, giao diện có thể hiện kiểu:

Query:
What is the capital of Asteria?

Chunk 0
Retrieval Risk:      0.31
Generation Risk:     0.12
Stealth Risk:        0.18
Pipeline Risk:       0.20
Overall:             0.20
SAFE

Chunk 3
Retrieval Risk:      0.81
Generation Risk:     0.92
Stealth Risk:        0.47
Pipeline Risk:       0.76
Overall:             0.79
SUSPICIOUS ⚠

rồi bên dưới:

Before Defense:
Nova City

Removed:
Chunk 3

After Defense:
Lumen City






Đấy sẽ là demo rất trực quan.
Tên sản phẩm mình thấy hợp nhất là RAGShield hoặc RAGPoisonGuard. Còn trong báo cáo nghiên cứu có thể gọi formal hơn là Four-Layer Defense Framework Against Robust RAG Poisoning.
Lưu ý quan trọng: file hiện tại là framework prototype có khả năng phân tích + loại chunk + regenerate, nhưng chưa được chứng minh là chống Confundo hiệu quả. Phần nghiên cứu tiếp theo chính là chạy Confundo thật rồi đo ASR before → ASR after, clean accuracy, FPR và ablation từng layer. Đây mới là phần biến code thành contribution đồ án.

