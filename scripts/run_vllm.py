"""Main experiment with vLLM: same models, prompts, tests and scoring as run_experiment.py, but fast.

Thinking ON, max 8,192 new tokens for every test. Decoding: greedy, or the article's sampling settings
(temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5). Output rows have the same fields as run_experiment.py.

Usage: python run_vllm.py --model M1 --sampling paper --seed 0 --n 100 --tag n100_paper_8k_s0
"""
import argparse
import json
import time

from vllm import LLM, SamplingParams

from common import ROOT, gsm_correct, has_loop, jload, lang_of, repeat_share
from run_experiment import MATH_INSTR, MODELS, REVISIONS, split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["M0", "M1"], required=True)
    p.add_argument("--sampling", choices=["greedy", "paper"], default="paper")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max_new", type=int, default=8192)
    p.add_argument("--tests", default="math_en,math_sv,general_sv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    t0 = time.time()
    out = ROOT / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    llm = LLM(MODELS[args.model], revision=REVISIONS[args.model], max_model_len=args.max_new + 1024,
              gpu_memory_utilization=0.85, seed=args.seed)
    tok = llm.get_tokenizer()
    sp = (SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=args.max_new, seed=args.seed)
          if args.sampling == "paper" else SamplingParams(temperature=0, max_tokens=args.max_new))

    problems = jload("gsm8k_sv_test.jsonl")[: args.n]
    questions = jload("test_questions_sv.json")[: args.n]
    tests = {"math_en": [f"{x['question_en']}\n{MATH_INSTR['en']}" for x in problems],
             "math_sv": [f"{x['question_sv']}\n{MATH_INSTR['sv']}" for x in problems],
             "general_sv": questions}
    for name in args.tests.split(","):
        idx = list(range(args.start, min(args.n, len(tests[name]))))
        prompts = [tok.apply_chat_template([{"role": "user", "content": tests[name][i]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) for i in idx]
        res = llm.generate(prompts, sp)
        with open(out / f"{args.model}_{name}.jsonl", "w") as f:
            for i, r in zip(idx, res):
                o = r.outputs[0]
                think, closed, answer = split(o.text)
                finished = o.finish_reason == "stop" and closed and bool(answer)
                row = {"i": i, "decoding": args.sampling, "max_new": args.max_new,
                       "prompt": tests[name][i], "finished": finished, "stuck": not finished,
                       "think_repeat_share": repeat_share(think), "think_loop": has_loop(think),
                       "n_tokens": len(o.token_ids), "answer": answer if finished else "", "text": o.text}
                if name.startswith("math"):
                    row["gold"] = problems[i]["answer"]
                    row["correct"] = finished and gsm_correct(o.text, problems[i]["answer"])
                else:
                    row["answer_lang"] = lang_of(answer) if finished else None
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(args.model, args.sampling, name, f"{len(idx)} items done [{time.time() - t0:.0f}s]", flush=True)
    meta = {"model": args.model, "hf_id": MODELS[args.model], "revision": REVISIONS[args.model], "decoding": args.sampling,
            "thinking": True, "max_new_tokens": args.max_new, "n": args.n, "seed": args.seed, "engine": "vllm"}
    (out / f"{args.model}_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
