"""M0 vs M1 table from whatever answers exist so far, one block per setting (results/<tag>/*.jsonl, synced from arr).
Every row states decoding and context length. Writes results/progress.md."""
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
SETTINGS = [("n100_greedy_8k", "greedy, max 8,192 tokens"),
            ("n100_paper_8k_s0", "article sampling (seed 0), max 8,192 tokens")]
TESTS = [("math_en", "English math"), ("math_sv", "Swedish math"), ("general_sv", "Swedish general questions")]


def load(tag, m, t):
    f = RES / tag / f"{m}_{t}.jsonl"
    return {r["i"]: r for r in map(json.loads, f.read_text().splitlines())} if f.exists() else {}


def judge(tag):
    f = RES / "judge" / f"{tag}_general_score.json"
    return json.loads(f.read_text()) if f.exists() else None


HEAD = ["| Test | Items | M0 finished | M1 finished | M0 correct | M1 correct | M0 avg repetition | M1 avg repetition |",
        "|---|---|---|---|---|---|---|---|"]
blocks = []
for tag, setting in SETTINGS:
    if not any((RES / tag).glob("*.jsonl")):
        continue
    items = "items 0-99"
    rows = [f"**{setting.capitalize()}, {items}**", ""] + HEAD
    for t, label in TESTS:
        A, B = load(tag, "M0", t), load(tag, "M1", t)
        common = sorted(set(A) & set(B))
        n = len(common)
        if not n:
            rows.append(f"| {label} | running | | | | | | |")
            continue
        fin = [sum(D[i]["finished"] for i in common) for D in (A, B)]
        rep = [100 * sum(D[i]["think_repeat_share"] for i in common) / n for D in (A, B)]
        if t.startswith("math"):
            cor = [str(sum(D[i]["correct"] for i in common)) for D in (A, B)]
        else:
            j = judge(tag)
            cor = [str(j["sensible_a"]), str(j["sensible_b"])] if j else ["judge pending"] * 2
        rows.append(f"| {label} | {n} | {fin[0]} | {fin[1]} | {cor[0]} | {cor[1]} | {rep[0]:.0f}% | {rep[1]:.0f}% |")
    blocks.append("\n".join(rows))
text = "\n\n".join(blocks) + ("\n\nThinking on for all tests. Finished = closed its thinking and gave an answer within the token limit. "
                          "Correct = right final number (math) or judged sensible by a blind A/B judge (general questions). "
                          "Repetition = share of the thinking text that repeats earlier text, averaged over items.\n")
(RES / "progress.md").write_text(text)
print(text)
