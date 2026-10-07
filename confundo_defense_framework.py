
"""
confundo_defense_framework.py

Research prototype with four clearly marked defense layers:

[1] Retrieval Defense
[2] Generation Defense
[3] Stealthiness Defense
[4] Pipeline / Fragmentation Defense

This is a research prototype, not a proven universal Confundo defense.
Thresholds and weights must be tuned on validation data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence, Optional, List, Dict, Any, Tuple
import math
import re

import numpy as np
from sentence_transformers import SentenceTransformer, util


AnswerFn = Callable[[str, Sequence[str]], str]


@dataclass
class ChunkReport:
    index: int
    chunk: str
    retrieval_risk: float = 0.0
    generation_risk: float = 0.0
    stealth_risk: float = 0.0
    pipeline_risk: float = 0.0
    overall_risk: float = 0.0
    suspicious: bool = False
    diagnostics: Optional[Dict[str, Any]] = None


@dataclass
class FrameworkResult:
    query: str
    original_answer: Optional[str]
    defended_answer: Optional[str]
    suspicious_indices: List[int]
    removed_chunks: List[str]
    kept_chunks: List[str]
    reports: List[ChunkReport]
    llm_calls: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "original_answer": self.original_answer,
            "defended_answer": self.defended_answer,
            "suspicious_indices": self.suspicious_indices,
            "removed_chunks": self.removed_chunks,
            "kept_chunks": self.kept_chunks,
            "reports": [asdict(r) for r in self.reports],
            "llm_calls": self.llm_calls,
        }


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _minmax(values: Sequence[float]) -> List[float]:
    vals = list(map(float, values))
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in vals]
    return [(x - lo) / (hi - lo) for x in vals]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 1.0
    return len(ta & tb) / len(union)


# ===========================================================================
# [LAYER 1] RETRIEVAL DEFENSE
# ===========================================================================
class RetrievalDefense:
    """
    Detect chunks that look unusually optimized to dominate retrieval.

    Signals:
    - query/chunk embedding similarity
    - positive similarity outlier z-score relative to peer chunks
    """

    def __init__(self, embedder: SentenceTransformer, z_scale: float = 1.5):
        self.embedder = embedder
        self.z_scale = z_scale

    def score(
        self,
        query: str,
        chunks: Sequence[str],
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        chunks = list(chunks)
        if not chunks:
            return [], []

        q_emb = self.embedder.encode(
            [query], convert_to_tensor=True, normalize_embeddings=True
        )
        c_emb = self.embedder.encode(
            chunks, convert_to_tensor=True, normalize_embeddings=True
        )

        sims = util.cos_sim(q_emb, c_emb)[0].detach().cpu().tolist()
        rel = _minmax(sims)
        mean = float(np.mean(sims))
        std = float(np.std(sims))

        risks, diagnostics = [], []

        for sim, relative in zip(sims, rel):
            z = (sim - mean) / std if std > 1e-12 else 0.0
            outlier = _sigmoid(max(0.0, z) * self.z_scale) if z > 0 else 0.0
            risk = _clip01(0.60 * relative + 0.40 * outlier)

            risks.append(risk)
            diagnostics.append({
                "query_similarity": float(sim),
                "relative_similarity": float(relative),
                "similarity_z": float(z),
                "retrieval_outlier": float(outlier),
            })

        return risks, diagnostics


# ===========================================================================
# [LAYER 2] GENERATION DEFENSE
# ===========================================================================
class GenerationDefense:
    """
    Leave-one-out generation influence defense.

    Does NOT need the attack target to compute risk.

    For each chunk:
    - generate answer with all chunks
    - remove the chunk
    - regenerate
    - compare semantic + lexical answer change

    target_answer is optional and only used as a diagnostic.
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        semantic_weight: float = 0.75,
        lexical_weight: float = 0.25,
    ):
        self.embedder = embedder
        total = semantic_weight + lexical_weight
        if total <= 0:
            raise ValueError("Generation weights must sum to > 0.")
        self.semantic_weight = semantic_weight / total
        self.lexical_weight = lexical_weight / total

    def _answer_similarity(self, a: str, b: str) -> float:
        emb = self.embedder.encode(
            [a, b], convert_to_tensor=True, normalize_embeddings=True
        )
        sim = float(util.cos_sim(emb[0], emb[1]).item())
        return max(-1.0, min(1.0, sim))

    def score(
        self,
        query: str,
        chunks: Sequence[str],
        answer_fn: AnswerFn,
        target_answer: Optional[str] = None,
    ) -> Tuple[List[float], List[Dict[str, Any]], str, int]:
        chunks = list(chunks)
        if not chunks:
            return [], [], "", 0

        llm_calls = 0
        original_answer = answer_fn(query, chunks)
        llm_calls += 1

        target_in_original = (
            bool(target_answer)
            and target_answer.lower() in original_answer.lower()
        )

        risks, diagnostics = [], []

        for i in range(len(chunks)):
            ablated = chunks[:i] + chunks[i + 1:]
            answer_without = answer_fn(query, ablated)
            llm_calls += 1

            sem_sim = self._answer_similarity(original_answer, answer_without)
            semantic_change = _clip01((1.0 - sem_sim) / 2.0)

            lexical_sim = _token_jaccard(original_answer, answer_without)
            lexical_change = _clip01(1.0 - lexical_sim)

            risk = _clip01(
                self.semantic_weight * semantic_change
                + self.lexical_weight * lexical_change
            )

            target_flip = False
            if target_answer:
                target_in_without = target_answer.lower() in answer_without.lower()
                target_flip = target_in_original and not target_in_without

            risks.append(risk)
            diagnostics.append({
                "answer_without_chunk": answer_without,
                "semantic_similarity": sem_sim,
                "semantic_change": semantic_change,
                "lexical_similarity": lexical_sim,
                "lexical_change": lexical_change,
                "target_flip_diagnostic": bool(target_flip),
            })

        return risks, diagnostics, original_answer, llm_calls


# ===========================================================================
# [LAYER 3] STEALTHINESS DEFENSE
# ===========================================================================
class StealthinessDefense:
    """
    Adds signals that natural-looking poison text may still fail to hide from.

    Signals:
    - semantic outlier relative to peer chunks
    - optional source/provenance trust
    - lightweight instruction/meta-text markers

    source_trust[i]:
      1.0 = highly trusted source
      0.0 = untrusted source
    """

    DEFAULT_MARKERS = (
        "ignore previous",
        "ignore all previous",
        "system prompt",
        "developer message",
        "follow these instructions",
        "answer:",
        "question:",
        "the user wants",
        "only the answer",
    )

    def __init__(
        self,
        embedder: SentenceTransformer,
        markers: Optional[Sequence[str]] = None,
    ):
        self.embedder = embedder
        self.markers = tuple(markers or self.DEFAULT_MARKERS)

    def score(
        self,
        chunks: Sequence[str],
        source_trust: Optional[Sequence[float]] = None,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        chunks = list(chunks)
        if not chunks:
            return [], []

        if source_trust is not None and len(source_trust) != len(chunks):
            raise ValueError("source_trust length must equal number of chunks.")

        emb = self.embedder.encode(
            chunks, convert_to_tensor=True, normalize_embeddings=True
        )
        sim_matrix = util.cos_sim(emb, emb).detach().cpu().numpy()

        risks, diagnostics = [], []

        for i, chunk in enumerate(chunks):
            peer_sims = [
                float(sim_matrix[i][j])
                for j in range(len(chunks))
                if j != i
            ]

            if peer_sims:
                mean_peer_sim = float(np.mean(peer_sims))
                semantic_outlier = _clip01((1.0 - mean_peer_sim) / 2.0)
            else:
                mean_peer_sim = 1.0
                semantic_outlier = 0.0

            lower = chunk.lower()
            marker_hits = sum(1 for marker in self.markers if marker in lower)
            marker_risk = _clip01(marker_hits / 2.0)

            if source_trust is not None:
                trust = _clip01(float(source_trust[i]))
                provenance_risk = 1.0 - trust
                risk = _clip01(
                    0.50 * provenance_risk
                    + 0.35 * semantic_outlier
                    + 0.15 * marker_risk
                )
            else:
                provenance_risk = 0.0
                risk = _clip01(
                    0.70 * semantic_outlier
                    + 0.30 * marker_risk
                )

            risks.append(risk)
            diagnostics.append({
                "mean_peer_similarity": mean_peer_sim,
                "semantic_outlier": semantic_outlier,
                "marker_risk": marker_risk,
                "provenance_risk": provenance_risk,
            })

        return risks, diagnostics


# ===========================================================================
# [LAYER 4] PIPELINE / FRAGMENTATION DEFENSE
# ===========================================================================
class PipelineDefense:
    """
    Detects content whose query relevance persists across several re-chunking
    settings, a useful supporting signal against fragmentation-robust poison.

    A clean concise fact can also score high, so this layer must not be used as
    the only blocker.
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        fragment_sizes: Sequence[int] = (8, 16, 32),
        similarity_threshold: float = 0.55,
    ):
        self.embedder = embedder
        self.fragment_sizes = tuple(max(2, int(x)) for x in fragment_sizes)
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _word_fragments(text: str, size: int) -> List[str]:
        words = text.split()
        if not words:
            return [""]
        if len(words) <= size:
            return [text]

        fragments = []
        step = max(1, size // 2)

        for start in range(0, len(words), step):
            frag = words[start:start + size]
            if not frag:
                break
            fragments.append(" ".join(frag))
            if start + size >= len(words):
                break

        return fragments

    def score(
        self,
        query: str,
        chunks: Sequence[str],
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        chunks = list(chunks)
        if not chunks:
            return [], []

        q_emb = self.embedder.encode(
            [query], convert_to_tensor=True, normalize_embeddings=True
        )

        risks, diagnostics = [], []

        for chunk in chunks:
            per_size_max = []

            for size in self.fragment_sizes:
                fragments = self._word_fragments(chunk, size)

                f_emb = self.embedder.encode(
                    fragments, convert_to_tensor=True, normalize_embeddings=True
                )

                sims = util.cos_sim(q_emb, f_emb)[0].detach().cpu().tolist()
                per_size_max.append(max(sims) if sims else 0.0)

            persistent_hits = sum(
                1 for x in per_size_max if x >= self.similarity_threshold
            )

            persistence = (
                persistent_hits / len(per_size_max)
                if per_size_max else 0.0
            )

            peak_sim = max(per_size_max, default=0.0)
            peak_risk = _clip01((peak_sim + 1.0) / 2.0)

            risk = _clip01(
                0.60 * persistence
                + 0.40 * peak_risk
            )

            risks.append(risk)
            diagnostics.append({
                "fragment_sizes": list(self.fragment_sizes),
                "max_similarity_by_size": per_size_max,
                "fragment_persistence": persistence,
                "fragment_peak_similarity": peak_sim,
            })

        return risks, diagnostics


# ===========================================================================
# MASTER FRAMEWORK — COMBINES ALL FOUR LAYERS
# ===========================================================================
class ConfundoDefenseFramework:
    """
    Combines:
      [1] Retrieval Defense
      [2] Generation Defense
      [3] Stealthiness Defense
      [4] Pipeline / Fragmentation Defense

    Default weights are research starting points, not universal constants.
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        embedding_device: str = "cpu",
        retrieval_weight: float = 0.25,
        generation_weight: float = 0.40,
        stealth_weight: float = 0.15,
        pipeline_weight: float = 0.20,
        overall_threshold: float = 0.60,
        max_remove: int = 1,
    ):
        self.embedder = SentenceTransformer(
            embedding_model,
            device=embedding_device,
        )

        self.retrieval = RetrievalDefense(self.embedder)
        self.generation = GenerationDefense(self.embedder)
        self.stealth = StealthinessDefense(self.embedder)
        self.pipeline = PipelineDefense(self.embedder)

        weights = [
            retrieval_weight,
            generation_weight,
            stealth_weight,
            pipeline_weight,
        ]

        if any(w < 0 for w in weights):
            raise ValueError("Framework weights must be non-negative.")

        total = sum(weights)
        if total <= 0:
            raise ValueError("At least one framework weight must be positive.")

        self.retrieval_weight = retrieval_weight / total
        self.generation_weight = generation_weight / total
        self.stealth_weight = stealth_weight / total
        self.pipeline_weight = pipeline_weight / total

        self.overall_threshold = overall_threshold
        self.max_remove = max(0, int(max_remove))

    def analyze_and_defend(
        self,
        query: str,
        chunks: Sequence[str],
        answer_fn: Optional[AnswerFn] = None,
        target_answer: Optional[str] = None,
        source_trust: Optional[Sequence[float]] = None,
    ) -> FrameworkResult:
        chunks = list(chunks)

        if not chunks:
            raise ValueError("chunks must not be empty.")

        n = len(chunks)

        # [LAYER 1]
        retrieval_risk, retrieval_diag = self.retrieval.score(query, chunks)

        # [LAYER 2]
        llm_calls = 0
        original_answer = None

        if answer_fn is not None:
            (
                generation_risk,
                generation_diag,
                original_answer,
                generation_calls,
            ) = self.generation.score(
                query,
                chunks,
                answer_fn,
                target_answer=target_answer,
            )
            llm_calls += generation_calls
        else:
            generation_risk = [0.0] * n
            generation_diag = [{"skipped": True} for _ in range(n)]

        # [LAYER 3]
        stealth_risk, stealth_diag = self.stealth.score(
            chunks,
            source_trust=source_trust,
        )

        # [LAYER 4]
        pipeline_risk, pipeline_diag = self.pipeline.score(query, chunks)

        reports: List[ChunkReport] = []

        for i, chunk in enumerate(chunks):
            overall = _clip01(
                self.retrieval_weight * retrieval_risk[i]
                + self.generation_weight * generation_risk[i]
                + self.stealth_weight * stealth_risk[i]
                + self.pipeline_weight * pipeline_risk[i]
            )

            suspicious = overall >= self.overall_threshold

            reports.append(
                ChunkReport(
                    index=i,
                    chunk=chunk,
                    retrieval_risk=retrieval_risk[i],
                    generation_risk=generation_risk[i],
                    stealth_risk=stealth_risk[i],
                    pipeline_risk=pipeline_risk[i],
                    overall_risk=overall,
                    suspicious=suspicious,
                    diagnostics={
                        "retrieval": retrieval_diag[i],
                        "generation": generation_diag[i],
                        "stealthiness": stealth_diag[i],
                        "pipeline": pipeline_diag[i],
                    },
                )
            )

        candidates = [r for r in reports if r.suspicious]
        candidates.sort(key=lambda r: r.overall_risk, reverse=True)
        selected = candidates[: self.max_remove]

        suspicious_indices = sorted(r.index for r in selected)
        suspicious_set = set(suspicious_indices)

        kept_chunks = [
            chunk for i, chunk in enumerate(chunks)
            if i not in suspicious_set
        ]

        removed_chunks = [
            chunks[i] for i in suspicious_indices
        ]

        defended_answer = None

        if answer_fn is not None:
            defended_answer = answer_fn(query, kept_chunks)
            llm_calls += 1

        return FrameworkResult(
            query=query,
            original_answer=original_answer,
            defended_answer=defended_answer,
            suspicious_indices=suspicious_indices,
            removed_chunks=removed_chunks,
            kept_chunks=kept_chunks,
            reports=reports,
            llm_calls=llm_calls,
        )


def print_defense_report(result: FrameworkResult) -> None:
    print("=" * 78)
    print("CONFUNDO FOUR-LAYER DEFENSE REPORT")
    print("=" * 78)

    if result.original_answer is not None:
        print("\nOriginal answer:")
        print(result.original_answer)

    print("\nPer-chunk risks:")

    for r in result.reports:
        mark = "  <-- FLAGGED" if r.index in result.suspicious_indices else ""

        print(
            f"\nChunk {r.index}{mark}\n"
            f"  [1] Retrieval Defense    : {r.retrieval_risk:.4f}\n"
            f"  [2] Generation Defense   : {r.generation_risk:.4f}\n"
            f"  [3] Stealthiness Defense : {r.stealth_risk:.4f}\n"
            f"  [4] Pipeline Defense      : {r.pipeline_risk:.4f}\n"
            f"      OVERALL RISK          : {r.overall_risk:.4f}\n"
            f"      Text                  : {r.chunk}"
        )

    print("\nRemoved chunk indices:")
    print(result.suspicious_indices)

    if result.defended_answer is not None:
        print("\nDefended answer:")
        print(result.defended_answer)

    print("\nLLM calls:")
    print(result.llm_calls)


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================
#
# guard = ConfundoDefenseFramework(
#     embedding_device="cpu",
#     overall_threshold=0.60,
#     max_remove=1,
# )
#
# result = guard.analyze_and_defend(
#     query=query,
#     chunks=retrieved_chunks,
#     answer_fn=my_rag_answer_function,
#     target_answer=None,   # optional, evaluation only
#     source_trust=None,    # optional list in [0,1]
# )
#
# print_defense_report(result)
