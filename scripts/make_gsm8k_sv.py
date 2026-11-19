"""Build a small Swedish GSM8K by machine-translating problems with opus-mt-en-sv.

dev  = first 60 GSM8K *train* problems (for choosing edit settings)
test = first 150 GSM8K *test* problems (final numbers only)
rep  = GSM8K *test* problems 150-399 (a fresh replication set, added after the main test runs)
A problem is kept only if every number in the English text survives translation.
"""
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MT = "Helsinki-NLP/opus-mt-en-sv"
NUM = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text):
    return sorted(n.replace(",", "") for n in NUM.findall(text))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT).to(device).eval()

    def translate(texts):
        out = []
        for i in range(0, len(texts), 16):
            batch = tok(texts[i : i + 16], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                gen = model.generate(**batch, num_beams=4, max_new_tokens=400)
            out += tok.batch_decode(gen, skip_special_tokens=True)
        return out

    ds = load_dataset("openai/gsm8k", "main")
    import sys
    splits = [("dev", "train", 0, 60), ("test", "test", 0, 150), ("rep", "test", 150, 400)]
    if len(sys.argv) > 1:
        splits = [x for x in splits if x[0] in sys.argv[1:]]
    for split, src, lo, n in splits:
        rows = ds[src].select(range(lo, n))
        sv = translate(list(rows["question"]))
        kept = []
        for i, (r, q_sv) in enumerate(zip(rows, sv)):
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            # Swedish writes decimals with a comma; normalise before comparing number sets
            if numbers(r["question"]) != numbers(q_sv.replace(" ", "")) and numbers(r["question"]) != numbers(q_sv):
                continue
            kept.append({"id": f"gsm8k-{src}-{lo + i}", "question_en": r["question"], "question_sv": q_sv, "answer": gold})
        path = ROOT / f"data/gsm8k_sv_{split}.jsonl"
        path.write_text("\n".join(json.dumps(k, ensure_ascii=False) for k in kept) + "\n")
        print(f"{split}: kept {len(kept)}/{n - lo} -> {path}")


if __name__ == "__main__":
    main()
