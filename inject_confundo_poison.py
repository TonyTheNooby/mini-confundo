import json


CLEAN_FILE = "data/clean_docs.json"

POISON_FILE = "data/generated_poison.txt"

OUTPUT_FILE = "data/confundo_poisoned_docs.json"


# =========================
# LOAD CLEAN DOCUMENTS
# =========================

with open(
    CLEAN_FILE,
    "r",
    encoding="utf-8"
) as f:

    documents = json.load(f)


print(
    f"[+] Loaded {len(documents)} clean documents"
)


# =========================
# LOAD GENERATED POISON
# =========================

with open(
    POISON_FILE,
    "r",
    encoding="utf-8"
) as f:

    poison = f.read().strip()


print("\n[+] Confundo poison:")
print(poison)


# =========================
# INSERT POISON
# =========================

documents.append({
    "id": "doc_poison",
    "text": poison
})


# =========================
# SAVE
# =========================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        documents,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"\n[+] Created {OUTPUT_FILE}"
)

print(
    f"[+] Total documents: {len(documents)}"
)
