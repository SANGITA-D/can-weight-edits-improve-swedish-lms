"""Shared helpers: loading, batched generation, metrics, weight edits. Used by every experiment script."""
import json
import re
from collections import Counter
from pathlib import Path

import sacrebleu
import torch
from langdetect import DetectorFactory, detect
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

DetectorFactory.seed = 0
ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
FLORES = ROOT / "data/flores/flores200_dataset"
TRANSLATE_PROMPT = "Translate the following English text into Swedish. Reply with the Swedish translation only.\n\n{src}"


# ---------------------------------------------------------------- loading / generation
def load(model_id, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(DEVICE).eval()
    return tok, model


def chat(tok, user_msgs, system=None, enable_thinking=None):
    out = []
    for m in user_msgs:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": m}]
        kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
        out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))
    return out


def stop_ids(tok, model):
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids])
    ids |= {tok.convert_tokens_to_ids(t) for t in ["<|im_end|>", "<|endoftext|>"] if t in tok.get_vocab()}
    return ids


class PresencePenalty:
    """vLLM-style presence penalty: subtract `penalty` from the logit of every token already generated
    (prompt tokens are not penalised). Used to match the paper's evaluation sampling."""

    def __init__(self, penalty):
        self.penalty, self.prompt_len = penalty, None

    def __call__(self, input_ids, scores):
        if self.prompt_len is None:
            self.prompt_len = input_ids.shape[1]
        gen = input_ids[:, self.prompt_len :]
        if gen.shape[1]:
            seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
            scores = scores - self.penalty * seen.to(scores.dtype)
        return scores


PAPER_SAMPLING = {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20}  # + presence penalty 1.5


@torch.no_grad()
def generate(tok, model, prompts, max_new_tokens=256, batch_size=16, **gen_kwargs):
    """Returns list of dicts: text (special tokens stripped), finished (stopped before the cap), n_tokens."""
    stops = stop_ids(tok, model)
    gen_kwargs.setdefault("do_sample", False)
    presence = gen_kwargs.pop("presence_penalty", None)
    res = []
    for i in range(0, len(prompts), batch_size):
        b = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(DEVICE)
        extra = {"logits_processor": LogitsProcessorList([PresencePenalty(presence)])} if presence else {}
        out = model.generate(**b, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **gen_kwargs, **extra)
        for row in out[:, b["input_ids"].shape[1] :].tolist():
            cut = next((k for k, t in enumerate(row) if t in stops), None)
            body = row if cut is None else row[:cut]
            res.append({"text": tok.decode(body, skip_special_tokens=True), "finished": cut is not None, "n_tokens": len(body)})
    return res


# ---------------------------------------------------------------- metrics
def lang_of(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"


def has_loop(text, n=4, min_count=3):
    """True if any word 4-gram occurs >= 3 times: the model is repeating itself."""
    words = text.lower().split()
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return any(c >= min_count for c in grams.values())


def repeat_share(text, n=8):
    """Share of word 8-grams that already appeared earlier in the same text (0 = no repetition, ~1 = stuck)."""
    w = text.lower().split()
    seen, rep, total = set(), 0, 0
    for i in range(len(w) - n + 1):
        g = tuple(w[i : i + n])
        rep += g in seen
        total += 1
        seen.add(g)
    return rep / total if total else 0.0


def distinct2(text):
    """Share of word bigrams that are unique (1.0 = no repetition)."""
    w = text.lower().split()
    bi = list(zip(w, w[1:]))
    return len(set(bi)) / len(bi) if bi else 1.0


def no_da_rate(text):
    """Danish/Norwegian letters (ae, o-slash) per 1000 characters. Swedish uses a-umlaut/o-umlaut instead."""
    return 1000 * sum(text.count(c) for c in "æøÆØ") / max(1, len(text))


# Danish/Norwegian words that are not normal Swedish. Calibrated on human Swedish (FLORES devtest, 20k words):
# only 'er' (8) and 'et' (1) occur there, i.e. ~0.45 hits per 1000 words of real Swedish.
NO_DA_WORDS = set("ikke jeg hvad hva hvordan hvor hvorfor hvilken hvilke hvis meget mye også være nogen noen nogle "
                  "gjøre gøre bliver kun er et af deg seg".split())


def no_da_words(text):
    """Danish/Norwegian marker words per 1000 words."""
    w = re.findall(r"[a-zåäöæø]+", text.lower())
    return 1000 * sum(x in NO_DA_WORDS for x in w) / max(1, len(w))


def strip_thinking(text):
    return text.split("</think>")[-1]


NUM = re.compile(r"-?\d[\d\s.,]*")


def last_number(text):
    """Last number in the answer; handles '1 000', '1,000' and Swedish decimal comma '2,5'."""
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    cands = NUM.findall(boxed[-1] if boxed else text)
    for c in reversed(cands):
        c = c.strip().rstrip(".,").replace(" ", "").replace("\u00a0", "")
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+", c):
            c = c.replace(",", "")
        c = c.replace(",", ".")
        try:
            v = float(c)
            return int(v) if v.is_integer() else v
        except ValueError:
            continue
    return None


def gsm_correct(text, gold):
    pred = last_number(strip_thinking(text))
    return pred is not None and abs(float(pred) - float(gold)) < 1e-6


@torch.no_grad()
def perplexity(tok, model, texts, batch_size=16):
    """Token-level perplexity of plain text (no chat template)."""
    nll, n = 0.0, 0
    tok.padding_side = "right"
    for i in range(0, len(texts), batch_size):
        b = tok(texts[i : i + batch_size], return_tensors="pt", padding=True).to(DEVICE)
        logits = model(**b).logits[:, :-1].float()
        tgt = b["input_ids"][:, 1:]
        mask = b["attention_mask"][:, 1:].bool()
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll -= lp[mask].sum().item()
        n += mask.sum().item()
    tok.padding_side = "left"
    return float(torch.exp(torch.tensor(nll / n)))


def chrf(hyps, refs):
    return sacrebleu.corpus_chrf(hyps, [refs]).score


def first_line(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[0] if lines else ""


# ---------------------------------------------------------------- data
def jload(name):
    p = ROOT / "data" / name
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return json.loads(p.read_text())


def flores(lang, split, start=0, n=100):
    ext = "dev" if split == "dev" else "devtest"
    return (FLORES / split / f"{lang}.{ext}").read_text().splitlines()[start : start + n]


# ---------------------------------------------------------------- weight edits
@torch.no_grad()
def project_out(weight, u, alpha=1.0):
    """W' = W - alpha * u (u^T W) for a Linear weight [d_model, d_in]: stops the layer writing along u."""
    u = (u / u.norm()).to(weight.dtype).to(weight.device)
    weight -= alpha * torch.outer(u, u @ weight)


def output_matrices(model, layers, which=("mlp", "attn")):
    """The matrices that write into the residual stream (MLP down_proj, attention o_proj)."""
    mats = []
    for l in layers:
        layer = model.model.layers[l]
        if "mlp" in which:
            mats.append((f"layers.{l}.mlp.down_proj", layer.mlp.down_proj.weight))
        if "attn" in which and hasattr(layer, "self_attn"):
            mats.append((f"layers.{l}.self_attn.o_proj", layer.self_attn.o_proj.weight))
    return mats


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
