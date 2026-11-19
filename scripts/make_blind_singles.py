"""Blind single-answer rating for the Swedish general questions (replaces A/B pairs, which let a judge mix up sides).

Only real answers are shown, shuffled together from M0 and M1, without model labels. Items where a model gave no answer
(stuck at the token limit) count as not sensible automatically.

Usage:  python make_blind_singles.py n100_paper_8k_s0            -> results/judge/<tag>_singles.jsonl + _singles_key.json
        python make_blind_singles.py n100_paper_8k_s0 --score    -> reads _singles_verdicts.jsonl, writes <tag>_general_score.json
Verdict lines: {"id": 0, "sensible": true|false, "note": "..."}
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "results/judge"


def rows(tag, m):
    return [json.loads(l) for l in (ROOT / f"results/{tag}/{m}_general_sv.jsonl").read_text().splitlines() if l.strip()]


def make(tag, seed=0):
    items = [(m, r) for m in ("M0", "M1") for r in rows(tag, m) if r["finished"] and r["answer"]]
    random.Random(seed).shuffle(items)
    J.mkdir(parents=True, exist_ok=True)
    (J / f"{tag}_singles.jsonl").write_text("".join(
        json.dumps({"id": k, "question": r["prompt"], "answer": r["answer"][:2000]}, ensure_ascii=False) + "\n"
        for k, (m, r) in enumerate(items)))
    (J / f"{tag}_singles_key.json").write_text(json.dumps({str(k): [m, r["i"]] for k, (m, r) in enumerate(items)}))
    print(f"wrote {len(items)} answers -> {J / (tag + '_singles.jsonl')}")


def score(tag):
    key = json.loads((J / f"{tag}_singles_key.json").read_text())
    v = [json.loads(l) for l in (J / f"{tag}_singles_verdicts.jsonl").read_text().splitlines() if l.strip()]
    assert sorted(x["id"] for x in v) == list(range(len(key))), "verdicts must cover every id exactly once"
    n = {m: len(rows(tag, m)) for m in ("M0", "M1")}
    sens = {"M0": 0, "M1": 0}
    for x in v:
        sens[key[str(x["id"])][0]] += bool(x["sensible"])
    res = {"tag": tag, "n_M0": n["M0"], "n_M1": n["M1"], "answered_M0": sum(k[0] == "M0" for k in key.values()),
           "answered_M1": sum(k[0] == "M1" for k in key.values()), "sensible_a": sens["M0"], "sensible_b": sens["M1"]}
    (J / f"{tag}_general_score.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tag")
    p.add_argument("--score", action="store_true")
    a = p.parse_args()
    score(a.tag) if a.score else make(a.tag)
