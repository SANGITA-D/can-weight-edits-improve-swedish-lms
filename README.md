# Can weight edits improve Swedish language models?

A critical look at *[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)* (Takehi et al., MIT CSAIL, 2026), and a test of whether its loop-stopping weight edit, found on English math, can be reused for Swedish out of the box.

- **Report:** [REPORT.pdf](REPORT.pdf) · the same text in Markdown: [REPORT.md](REPORT.md)
- **Lab notebook** (steps, results and decisions): [LAB_NOTES.md](LAB_NOTES.md)

## Main finding

- **M0** is the original small model (Qwen3.5-0.8B). With thinking on, it often gets stuck repeating itself and never answers.
- **M1** is the same model with the article's published edit, which is meant to stop this.
- M1 gets stuck far less often than M0 on **English math, the same problems in Swedish, and everyday Swedish questions**, with greedy decoding and with the article's sampling settings. It also gives more correct and more sensible answers.
- So an edit that fixes a reasoning fault, found on English math, carries over to **another language** and to **another kind of task**. Swedish models could likely reuse edits found on non-Swedish models; each edit should still be checked on the target Swedish model.

Results (100 items per test, thinking on, up to 8,192 tokens; correct = right number for math, sensible = rated by a blind AI judge for general questions):

| Setting | Test | M0 finished | M1 finished | M0 correct or sensible | M1 correct or sensible |
|---|---|---|---|---|---|
| Greedy | English math | 0 | 14 | 0 | 14 |
| Greedy | Swedish math | 2 | 35 | 2 | 25 |
| Greedy | Swedish general questions | 7 | 32 | 3 | 15 |
| Article's sampling | English math | 10 | 34 | 10 | 31 |
| Article's sampling | Swedish math | 41 | 77 | 34 | 52 |
| Article's sampling | Swedish general questions | 63 | 99 | 34 | 44 |

## Repository layout

```
REPORT.pdf, REPORT.md      the report (PDF and Markdown)
LAB_NOTES.md               lab notebook (steps, results, decisions)
data/gsm8k_sv_test.jsonl   GSM8K test problems in English and Swedish (machine-translated, checked;
                           hand-fixed items marked sv_fixed_by_hand); the first 100 are used
data/test_questions_sv.json  100 everyday Swedish questions written for this study
scripts/run_vllm.py        main experiment (vLLM): one model, one decoding setting, all 3 tests
scripts/run_experiment.py  the same experiment with plain transformers (CPU / laptop; slow)
scripts/progress.py        M0 vs M1 tables from the results
scripts/make_blind_singles.py  blind judge input (shuffled answers, no model labels) and scoring
scripts/make_pdf.py        REPORT.md -> REPORT.pdf
scripts/common.py          shared generation code and metrics
results/n100_greedy_8k/    every answer, greedy decoding
results/n100_paper_8k_s0/  every answer, the article's sampling settings (seed 0)
results/judge/             judge inputs, hidden keys, verdicts and scores
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # plus vllm==0.34.0 for run_vllm.py
cd scripts
for M in M0 M1; do
  python run_vllm.py --model $M --sampling greedy --n 100 --max_new 8192 --tag n100_greedy_8k
  python run_vllm.py --model $M --sampling paper --seed 0 --n 100 --max_new 8192 --tag n100_paper_8k_s0
done
python progress.py                                    # results tables
python make_blind_singles.py n100_paper_8k_s0         # judge input; verdicts -> results/judge/<tag>_singles_verdicts.jsonl
python make_blind_singles.py n100_paper_8k_s0 --score
```

## Models and data

| Item | Revision | Licence |
|---|---|---|
| Qwen/Qwen3.5-0.8B (M0) | 2fc06364715b967f1860aea9cf38778875588b17 | Apache-2.0 |
| RikiyaT/Qwen3.5-0.8B-weight-space-programming (M1, the article's edit) | d1671a178705bcab7644806d23c535517f35723b | Apache-2.0 |
| Helsinki-NLP/opus-mt-en-sv (first translation of the math problems) | cec444c9ac08a87fcfcd172f2e6be8e112fda53c | Apache-2.0 |
| openai/gsm8k | 740312add88f781978c0658806c59bc2815b9866 | MIT |

**Settings:** thinking on; up to 8,192 new tokens. Greedy decoding, and the article's sampling settings (temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, seed 0). vLLM 0.34, transformers 5.23.

## AI usage

Claude Code (Claude Opus 5.5) was used to set up and run all the tests of the hypotheses: it wrote the code, ran the experiments, and drafted the report and the lab notes. Claude subagents wrote and checked the Swedish questions and translations, judged the general answers blind (keys in `results/judge/`), and reviewed the report. I steered the experiments and ideas, and iterated on Claude's instructions as needed to keep the experiments clean, robust and reasonable. The article's editor model (GPT-6 Astra) was not used; its released edit was reused as published.
