import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


MODEL_PATH = "AlienHu/confundo-correctness"

OUTPUT_FILE = "data/generated_poison.txt"


print("[+] CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "[+] GPU:",
        torch.cuda.get_device_name(0)
    )


# =========================
# LOAD TOKENIZER
# =========================

print("[+] Loading Confundo tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


# =========================
# LOAD MODEL
# =========================

print("[+] Loading pretrained Confundo generator...")


if torch.cuda.is_available():

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

else:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model = model.to("cpu")


model.eval()

print("[+] Model loaded")


# =========================
# INPUT
# =========================

question = input(
    "\nTarget question: "
).strip()

target_answer = input(
    "Target answer: "
).strip()


# =========================
# CONFUNDO PROMPT
# =========================

prompt = (
    f"This is the question: '{question}', "
    f"and this is the target answer: "
    f"'{target_answer}'. "
    f"Please craft a short guiding corpus "
    f"to state this fact. "
    f"Directly output the crafted corpus "
    f"without any other words."
)


messages = [
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


model_inputs = tokenizer(
    [text],
    return_tensors="pt"
).to(model.device)


# =========================
# GENERATE POISON
# =========================

print("\n[+] Generating poison...")


with torch.no_grad():

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )


generated_ids = generated_ids[
    :,
    model_inputs.input_ids.shape[1]:
]


poison = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0].strip()


# =========================
# RESULT
# =========================

print("\n==============================")
print("CONFUNDO GENERATED POISON")
print("==============================")

print(poison)


# =========================
# SAVE
# =========================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(poison)


print(
    f"\n[+] Saved poison to {OUTPUT_FILE}"
)


if target_answer.lower() in poison.lower():

    print(
        "[+] Target answer appears in poison."
    )

else:

    print(
        "[!] WARNING: target answer "
        "was not found verbatim in poison."
    )
