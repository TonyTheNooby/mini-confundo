"""
Pipeline defense: chia chunk -> loại chunk lặp (giữ lại bằng chứng lặp) -> chấm điểm chunk còn lại.
Chạy tốt trên CPU 8GB: chỉ cần embedding nhỏ (all-MiniLM-L6-v2).

    from local_config import load_embedder
    from pipeline_defense import PipelineDefense
    emb = load_embedder()
    pd_ = PipelineDefense(lambda xs: emb.encode(xs, normalize_embeddings=True))
    result = pd_.run(docs={"doc1": text1, "doc2": text2}, query="...")
"""
import hashlib
import re
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Chunk:
    text: str
    doc_id: str
    idx: int
    emb: np.ndarray | None = None
    dup_count: int = 1                       # số chunk (kể cả chính nó) thuộc cùng cụm lặp
    dup_docs: set = field(default_factory=set)  # các doc có chunk trong cụm
    score: float = 0.0
    decision: str = "keep"                   # keep | quarantine | remove
    reasons: list = field(default_factory=list)


def split_chunks(text: str, doc_id: str, size: int = 60, overlap: int = 15):
    words = text.split()
    step = max(1, size - overlap)
    return [Chunk(" ".join(words[i:i + size]), doc_id, k)
            for k, i in enumerate(range(0, max(1, len(words)), step))
            if words[i:i + size]]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t.lower())).strip()


class PipelineDefense:
    def __init__(self, embed_fn, sim_thr=0.90, warn=0.35, remove=0.65,
                 w_rep=0.5, w_qsim=0.5):
        self.embed = embed_fn
        self.sim_thr, self.warn, self.remove = sim_thr, warn, remove
        self.w_rep, self.w_qsim = w_rep, w_qsim

    # ---- Bước 1: chia chunk
    def chunk(self, docs: dict):
        out = []
        for doc_id, text in docs.items():
            out += split_chunks(text, doc_id)
        embs = self.embed([c.text for c in out])
        for c, e in zip(out, embs):
            c.emb = np.asarray(e, dtype=np.float32)
        return out

    # ---- Bước 2: loại chunk lặp (exact -> near-duplicate), NHƯNG ghi lại mức lặp
    def dedup(self, chunks):
        reps, rep_embs, seen = [], [], {}
        for c in chunks:
            h = hashlib.md5(_norm(c.text).encode()).hexdigest()
            if h in seen:                                  # trùng hệt
                r = seen[h]
            elif rep_embs:                                 # trùng gần (cosine)
                sims = np.stack(rep_embs) @ c.emb
                j = int(np.argmax(sims))
                r = reps[j] if sims[j] >= self.sim_thr else None
            else:
                r = None
            if r is None:
                reps.append(c); rep_embs.append(c.emb); seen[h] = c
                c.dup_docs.add(c.doc_id)
            else:
                r.dup_count += 1
                r.dup_docs.add(c.doc_id)
        return reps

    # ---- Bước 3: đánh giá chunk còn lại
    def evaluate(self, reps, query: str):
        q = np.asarray(self.embed([query])[0], dtype=np.float32)
        sims = np.array([r.emb @ q for r in reps])
        med = np.median(sims)
        mad = np.median(np.abs(sims - med)) * 1.4826 + 1e-6
        for r, s in zip(reps, sims):
            rep = min(1.0, np.log2(r.dup_count) / 3.0)        # lặp >=8 lần -> 1.0
            z = (s - med) / mad
            qs = 1 / (1 + np.exp(-(z - 2.0)))                 # nổi bật bất thường so với các chunk khác
            r.score = self.w_rep * rep + self.w_qsim * qs
            if rep > 0:
                r.reasons.append(f"lặp x{r.dup_count} ({len(r.dup_docs)} doc)")
            if z > 2:
                r.reasons.append(f"giống query bất thường (z={z:.1f})")
            r.decision = ("remove" if r.score >= self.remove
                          else "quarantine" if r.score >= self.warn else "keep")
        return reps

    def run(self, docs: dict, query: str):
        chunks = self.chunk(docs)
        reps = self.evaluate(self.dedup(chunks), query)
        return {
            "total_chunks": len(chunks),
            "after_dedup": len(reps),
            "kept": [r for r in reps if r.decision == "keep"],
            "quarantined": [r for r in reps if r.decision == "quarantine"],
            "removed": [r for r in reps if r.decision == "remove"],
        }
