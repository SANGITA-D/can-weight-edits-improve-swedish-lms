"""Main experiment: does the article's loop-stopping weight edit work for Swedish out of the box?

Models
  M0  Qwen/Qwen3.5-0.8B                                (unedited)
  M1  RikiyaT/Qwen3.5-0.8B-weight-space-programming    (the article's published edit)

Tests (thinking ON, greedy decoding, the same items for both models)
  math_en     20 GSM8K word problems in English          max 4,096 new tokens
  math_sv     the same 20 problems in Swedish            max 4,096 new tokens
  general_sv  20 everyday questions in Swedish           max 2,048 new tokens

For every answer we record whether the model finished (closed its thinking and answered before the limit)
or got stuck, how repetitive its thinking was, and, for math, whether the final number is correct.
Answer quality for general_sv is judged afterwards by a blind A/B judge (make_blind_pairs.py / score_blind.py).

Usage: python run_experiment.py --model M0      (then --model M1)
"""
import argparse
import json
import time

import torch

from common import PAPER_SAMPLING, ROOT, chat, generate, gsm_correct, has_loop, jload, lang_of, load, repeat_share

MODELS = {"M0": "Qwen/Qwen3.5-0.8B", "M1": "RikiyaT/Qwen3.5-0.8B-weight-space-programming"}
REVISIONS = {"M0": "2fc06364715b967f1860aea9cf38778875588b17", "M1": "d1671a178705bcab7644806d23c535517f35723b"}
MATH_INSTR = {"en": "Please reason step by step, and put your final answer within \\boxed{}.",
              "sv": "Resonera steg för steg och skriv ditt slutliga svar inom \\boxed{}."}
OUT = ROOT / "results/main"  # overridden by --tag


def split(text):
    think, sep, answer = text.partition("</think>")
    return think, bool(sep), answer.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["M0", "M1"], required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max_math", type=int, default=4096)
    p.add_argument("--max_general", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=5)
    p.add_argument("--start", type=int, default=0, help="first item (to extend an earlier run)")
    p.add_argument("--tests", default="math_en,math_sv,general_sv")
    p.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="bfloat16 on the Mac GPU, float32 on CPU")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--machine", default="mac")
    p.add_argument("--sampling", choices=["greedy", "paper"], default="greedy",
                   help="paper = the article's settings: temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5")
    p.add_argument("--tag", default="", help="write results to results/<tag> instead of results/main")
    args = p.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    global OUT
    if args.tag:
        OUT = ROOT / "results" / args.tag
    gen_kw = dict(PAPER_SAMPLING, presence_penalty=1.5) if args.sampling == "paper" else {}
    if args.threads:
        torch.set_num_threads(args.threads)
    tok, model = load(MODELS[args.model], dtype=torch.float32 if args.dtype == "float32" else torch.bfloat16)
    summary = {"model": args.model, "hf_id": MODELS[args.model], "revision": REVISIONS[args.model],
               "decoding": args.sampling, "thinking": True, "n": args.n, "dtype": args.dtype, "machine": args.machine}
    problems = jload("gsm8k_sv_test.jsonl")[: args.n]
    questions = jload("test_questions_sv.json")[: args.n]

    tests = [("math_en", [f"{x['question_en']}\n{MATH_INSTR['en']}" for x in problems], args.max_math),
             ("math_sv", [f"{x['question_sv']}\n{MATH_INSTR['sv']}" for x in problems], args.max_math),
             ("general_sv", questions, args.max_general)]
    OUT.mkdir(parents=True, exist_ok=True)
    # Work in small rounds across all three tests, saving every answer at once, so partial results exist early.
    for start in range(args.start, args.n, args.chunk):
        for name, prompts, max_new in [t for t in tests if t[0] in args.tests.split(",")]:
            idx = list(range(start, min(start + args.chunk, args.n)))
            res = generate(tok, model, chat(tok, [prompts[i] for i in idx], enable_thinking=True),
                           max_new_tokens=max_new, batch_size=len(idx), **gen_kw)
            with open(OUT / f"{args.model}_{name}.jsonl", "a") as f:
                for i, r in zip(idx, res):
                    think, closed, answer = split(r["text"])
                    finished = r["finished"] and closed and bool(answer)
                    row = {"i": i, "machine": args.machine, "prompt": prompts[i], "finished": finished, "stuck": not finished,
                           "think_repeat_share": repeat_share(think), "think_loop": has_loop(think),
                           "n_tokens": r["n_tokens"], "answer": answer if finished else "", "text": r["text"]}
                    if name.startswith("math"):
                        row["gold"] = problems[i]["answer"]
                        row["correct"] = finished and gsm_correct(r["text"], problems[i]["answer"])
                    else:
                        row["answer_lang"] = lang_of(answer) if finished else None
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(args.model, name, f"items {idx[0]}-{idx[-1]} done", f"[{time.time() - t0:.0f}s]", flush=True)
    (OUT / f"{args.model}_meta.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
