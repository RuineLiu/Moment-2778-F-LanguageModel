import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "/Users/jorahmormont/PycharmProjects/Moment-2778-F-LanguageModel/train_model_2"
LOCAL_ONLY = os.getenv("HF_LOCAL_ONLY", "0") == "1"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.bfloat16 if device == "cuda" else (torch.float16 if device == "mps" else torch.float32)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    local_files_only=LOCAL_ONLY,
)
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=LOCAL_ONLY,
        device_map="auto" if device == "cuda" else None,
    )
except TypeError:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=LOCAL_ONLY,
        device_map="auto" if device == "cuda" else None,
    )
if device != "cuda":
    base_model = base_model.to(device)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

while True:
    q = input("你: ").strip()
    if q.lower() in {"exit", "quit", "q"}:
        break

    messages = [{"role": "user", "content": q}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    print("Raymond:", tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True))
