import sys
import json
import numpy as np
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# CONFIG
# =========================

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "Qwen/Qwen3-0.6B"

CHUNK_SIZE = 128
CHUNK_OVERLAP = 20
TOP_K = 3


# =========================
# 1. LOAD DOCUMENTS
# =========================

DATA_FILE = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "data/clean_docs.json"
)

print(f"[+] Dataset: {DATA_FILE}")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)


# =========================
# 2. CHUNKING
# =========================

chunk_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)


def chunk_text(text):
    token_ids = chunk_tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    start = 0

    while start < len(token_ids):

        end = min(
            start + CHUNK_SIZE,
            len(token_ids)
        )

        chunk_ids = token_ids[start:end]

        chunk = chunk_tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        )

        chunks.append(chunk)

        if end == len(token_ids):
            break

        start = end - CHUNK_OVERLAP

    return chunks


chunks = []

for doc in documents:

    doc_chunks = chunk_text(doc["text"])

    for i, text in enumerate(doc_chunks):

        chunks.append({
            "doc_id": doc["id"],
            "chunk_id": i,
            "text": text
        })


print(f"[+] Created {len(chunks)} chunks")


# =========================
# 3. EMBEDDING
# =========================

print("[+] Loading embedding model...")

# CPU deliberately:
# save RTX VRAM for Qwen
embedder = SentenceTransformer(
    EMBED_MODEL,
    device="cpu"
)

texts = [
    c["text"]
    for c in chunks
]

embeddings = embedder.encode(
    texts,
    normalize_embeddings=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

print("[+] Embedding shape:", embeddings.shape)


# =========================
# 4. FAISS INDEX
# =========================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(
    f"[+] FAISS contains {index.ntotal} vectors"
)


# =========================
# 5. RETRIEVAL
# =========================

def retrieve(question):

    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    k = min(TOP_K, len(chunks))

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx, score in zip(
        indices[0],
        scores[0]
    ):

        results.append({
            "score": float(score),
            **chunks[idx]
        })

    return results

# =========================
# 6. LOAD QWEN
# =========================

print("[+] Loading Qwen...")

cuda_available = torch.cuda.is_available()

dtype = (
    torch.float16
    if cuda_available
    else torch.float32
)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=dtype,
    device_map="auto" if cuda_available else None
)

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL
)

if not cuda_available:
    model = model.to("cpu")

print(
    "[+] LLM device:",
    model.device
)


# =========================
# 7. GENERATION
# =========================

def answer_question(question):

    retrieved = retrieve(question)

    print("\n===== RETRIEVAL =====")

    for rank, r in enumerate(
        retrieved,
        start=1
    ):

        print(
            f"\nRank {rank}"
            f"\nScore: {r['score']:.4f}"
            f"\nDocument: {r['doc_id']}"
            f"\nText: {r['text']}"
        )

    context = "\n\n".join(
        [
            f"[Document {i + 1}]\n{r['text']}"
            for i, r in enumerate(retrieved)
        ]
    )

    prompt = f"""
CONTEXT:

{context}

QUESTION:

{question}

Answer using only the information contained in CONTEXT.
If the answer cannot be found in CONTEXT, answer exactly:

NOT_FOUND

Keep the answer short.
"""

    messages = [
        {
            "role": "system",
            "content":
            "You are a retrieval-augmented question answering system."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    print("\n===== ANSWER =====")
    print(answer.strip())


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    while True:

        print("\n")
        question = input(
            "Question (type 'exit' to quit): "
        )

        if question.lower() == "exit":
            break

        answer_question(question)




