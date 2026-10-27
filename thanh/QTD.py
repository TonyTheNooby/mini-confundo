import os
import re
import math
import glob
import json
import random
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# BƯỚC 1: TẢI & LỌC TẬP CÂU HỎI Q_CAL TỪ HUGGING FACE (vapit/HarryPotterQA)
# =====================================================================
def get_clean_q_cal(
    hf_dataset_name: str = "vapit/HarryPotterQA",
    attack_questions: list = None,
    similarity_threshold: float = 0.75,
    sample_size: int = 200,
    seed: int = 42
) -> list:
    print(f"[1/5] Đang tải tập dữ liệu '{hf_dataset_name}' từ Hugging Face...")
    
    # 1. Tải dataset từ Hugging Face với ID "vapit/HarryPotterQA"
    dataset = load_dataset(hf_dataset_name, split="train")
    
    # 2. Trích xuất danh sách câu hỏi (kiểm tra linh hoạt cả "question" và "Question")
    raw_questions = []
    for item in dataset:
        q_text = item.get("question") or item.get("Question") or item.get("query")
        if q_text:
            raw_questions.append(str(q_text).strip())
            
    print(f" -> Tổng số câu hỏi lấy từ '{hf_dataset_name}': {len(raw_questions):,} câu.")

    # 3. Loại bỏ các câu trùng lặp chính xác
    unique_questions = list(dict.fromkeys([q for q in raw_questions if q]))
    print(f" -> Sau khi loại bỏ trùng lặp: {len(unique_questions):,} câu.")

    # 4. Rút ngẫu nhiên 200 câu hỏi hiệu chỉnh Q_cal
    random.seed(seed)
    Q_cal = random.sample(clean_questions, min(sample_size, len(clean_questions)))
    print(f" -> Đã chọn thành công |Q_cal| = {len(Q_cal)} câu hỏi hiệu chỉnh sạch.")
    
    return Q_cal

# =====================================================================
# BƯỚC 2: TẠO CORPUS SẠCH TỪ 7 FILE HARRY POTTER
# =====================================================================
def build_clean_corpus_from_sources(folder_path: str = None, chunk_size: int = 500, overlap: int = 50):
    # Nếu không truyền folder_path, tự động lấy thư mục chứa file .py hiện tại
    if folder_path is None:
        folder_path = os.path.dirname(os.path.abspath(__file__))
        
    print(f"Đang quét tìm file văn bản trong thư mục: {os.path.abspath(folder_path)}")
    
    # Quét tất cả file .txt trong thư mục
    files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    
    # Nếu không thấy file .txt, thử tìm file .markdown
    if not files:
        files = sorted(glob.glob(os.path.join(folder_path, "*.markdown")))
        
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file .txt hay .markdown nào trong thư mục: {os.path.abspath(folder_path)}")
        
    print(f" -> Đã tìm thấy {len(files)} file sách:")
    for f in files:
        print(f"    + {os.path.basename(f)}")
    
    all_chunks = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            
        step = chunk_size - overlap
        for i in range(0, len(text), step):
            chunk_text = text[i:i+chunk_size].strip()
            if len(chunk_text) >= 100:  # Lọc bỏ đoạn văn quá ngắn
                all_chunks.append(chunk_text)
                
    print(f" -> Tạo thành công {len(all_chunks):,} chunks sạch từ {len(files)} file.")
    return all_chunks

# =====================================================================
# BƯỚC 3: TRUY XUẤT TOP-50 CHUNK MỖI CÂU (n = 200 x 50 = 10,000 CHUNKS)
# =====================================================================
def retrieve_top50_per_query(Q_cal: list, corpus_chunks: list):
    print(f"\n[3/5] Đang truy xuất Top-50 chunk cho {len(Q_cal)} câu hỏi...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50000)
    corpus_vectors = vectorizer.fit_transform(corpus_chunks)
    q_vectors = vectorizer.transform(Q_cal)
    
    sim_matrix = cosine_similarity(q_vectors, corpus_vectors)
    retrieved_pairs = []  # Chứa n = 10,000 cặp (query, chunk)
    
    for q_idx, q_text in enumerate(Q_cal):
        scores = sim_matrix[q_idx]
        top50_indices = np.argsort(-scores)[:50]
        for c_idx in top50_indices:
            retrieved_pairs.append({
                "query": q_text,
                "chunk": corpus_chunks[c_idx]
            })
            
    print(f" -> Thu được n = {len(retrieved_pairs):,} chunk sạch ({len(Q_cal)} câu × 50 chunks).")
    return retrieved_pairs, vectorizer


# =====================================================================
# BƯỚC 4: TÍNH CHỈ SỐ QTD = QTD-IDF + QTD-SEM
# =====================================================================
def tokenize(text: str) -> list:
    return [t for t in re.findall(r'\b\w+\b', text.lower()) if len(t) > 1]

def build_idf_dict(corpus_chunks: list) -> dict:
    N = len(corpus_chunks)
    df = {}
    for chunk in corpus_chunks:
        for token in set(tokenize(chunk)):
            df[token] = df.get(token, 0) + 1
    return {token: math.log((N + 1) / (freq + 1)) + 1.0 for token, freq in df.items()}

def compute_qtd_score(q_toks: list, c_toks: list, idf_dict: dict, vectorizer, theta: float = 0.6) -> float:
    chunk_len = len(c_toks)
    if chunk_len == 0:
        return 0.0
    
    # 1. QTD-IDF = sum_{w in W(q)} (IDF(w) * tf(w, c)) / |c|
    tf_dict = {}
    for t in c_toks:
        tf_dict[t] = tf_dict.get(t, 0) + 1
    qtd_idf = sum(tf_dict.get(w, 0) * idf_dict.get(w, 1.0) for w in set(q_toks)) / chunk_len
    
    # 2. QTD-SEM = sum_{v in tokens(c)} 1[ max_{w in W(q)} cos(E(v), E(w)) > theta ] / |c|
    v_words = [v for v in c_toks if v in vectorizer.vocabulary_]
    q_words = [w for w in set(q_toks) if w in vectorizer.vocabulary_]
    if not v_words or not q_words:
        qtd_sem = 0.0
    else:
        v_vecs = vectorizer.transform(v_words)
        q_vecs = vectorizer.transform(q_words)
        sim_mat = cosine_similarity(v_vecs, q_vecs)
        qtd_sem = float(np.sum(sim_mat.max(axis=1) > theta) / chunk_len)
        
    return qtd_idf + qtd_sem


# =====================================================================
# BƯỚC 5: LẬP DANH SÁCH CAL VÀ TÍNH NGƯỠNG TAU (k = 500)
# =====================================================================
def calibrate_qtd_tau(retrieved_pairs: list, corpus_chunks: list, vectorizer, alpha: float = 0.05, theta: float = 0.6):
    print(f"\n[4/5] Đang tính IDF từ vựng trên toàn bộ corpus...")
    idf_dict = build_idf_dict(corpus_chunks)
    
    print(f"[5/5] Đang tính chỉ số QTD cho n = {len(retrieved_pairs):,} mẫu...")
    CAL = []
    for pair in retrieved_pairs:
        q_toks = tokenize(pair["query"])
        c_toks = tokenize(pair["chunk"])
        score = compute_qtd_score(q_toks, c_toks, idf_dict, vectorizer, theta=theta)
        CAL.append(score)
        
    n = len(CAL) # n = 10,000
    
    # Sắp xếp danh sách CAL theo thứ tự giảm dần
    CAL.sort(reverse=True)
    
    # Tính k = floor(alpha * (n + 1)) = floor(0.05 * 10001) = 500
    k = math.floor(alpha * (n + 1))
    
    # Lấy tau = giá trị đứng thứ k trong CAL (vị trí chỉ số k - 1)
    tau = CAL[k - 1]
    
    return tau, k, n, CAL


# =====================================================================
# CHẠY QUY TRÌNH & LƯU KẾT QUẢ
# =====================================================================
if __name__ == "__main__":
    # 1. Tải và lọc 200 câu hỏi Q_cal từ Hugging Face
    Q_cal = get_clean_q_cal(hf_dataset_name="vapit/harrypotterQA", sample_size=200, seed=42)
    
    # 2. Tạo Corpus sạch từ 7 file Harry Potter
    corpus_chunks = build_clean_corpus_from_sources()
    
    # 3. Truy xuất Top-50 chunk cho mỗi câu hỏi
    retrieved_pairs, vectorizer = retrieve_top50_per_query(Q_cal, corpus_chunks)
    
    # 4. Tính toán điểm QTD và trích xuất ngưỡng tau
    tau, k, n, CAL = calibrate_qtd_tau(retrieved_pairs, corpus_chunks, vectorizer, alpha=0.05, theta=0.6)
    
    # 5. Lưu kết quả duy nhất của con số tau vào tệp JSON
    output_result = {
        "metric": "Query-Text Dependence (QTD)",
        "dataset": "vapit/harrypotterQA",
        "num_queries_Q_cal": len(Q_cal),
        "num_chunks_n": n,
        "alpha": 0.05,
        "k_rank": k,
        "tau_threshold": round(tau, 6)
    }
    
    with open("calibrated_tau_qtd.json", "w", encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False, indent=2)
        
    print("\n==========================================================")
    print(" KẾT QUẢ HIỆU CHỈNH NGƯỠNG QTD (OFFLINE CALIBRATION):")
    print("==========================================================")
    print(f" -> Tổng số mẫu (n)            : {n:,}")
    print(f" -> Mức ý nghĩa (alpha)        : 0.05")
    print(f" -> Vị trí thứ k = ⌊α(n+1)⌋     : {k} (lớn thứ 500)")
    print(f" -> NGƯỠNG HIỆU CHỈNH (tau)    : {tau:.6f}")
    print("==========================================================")
    print("Đã lưu kết quả duy nhất vào tệp 'calibrated_tau_qtd.json'")