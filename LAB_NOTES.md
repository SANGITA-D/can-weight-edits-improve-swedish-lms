# Lab notebook

A short record of the steps, results and decisions behind this study. Times are CET, 19 November 2026.
**M0** = Qwen/Qwen3.5-0.8B (original); **M1** = RikiyaT/Qwen3.5-0.8B-weight-space-programming (the article's published edit).

## 1. Design

- **Question:** does the article's loop-stopping edit, found on English math, also help in Swedish, and on questions beyond math?
- **Tests (same items for both models):** English math (GSM8K test), the same problems in Swedish, and everyday Swedish questions. Thinking always on, since that is where the edit has something to fix.
- **Measures:** finished (closed the thinking and answered within the token limit); correct final number (math); sensible answer (general questions, blind AI judge); share of repeated 8-word sequences in the thinking.
- **Earlier attempts (not reported):** first runs with greedy decoding and a 4,096-token limit (2,048 for questions) on 20 items. Almost no answers finished for either model, so the limit was too short to show anything. A quick test with the article's sampling settings and 8,192 tokens on 5 English math problems gave a clear signal (M0 finished 0/5, M1 1/5; repetition 54% vs 11%), so all tests moved to 8,192 tokens.

## 2. Data

- **Swedish math:** GSM8K test problems machine-translated with Helsinki-NLP/opus-mt-en-sv, then checked against the English by AI. Problems with wrong units (e.g. "miles" became the Swedish "mil" = 10 km), lost sentences or wrong words were re-translated by hand, keeping every number (marked `sv_fixed_by_hand`). Items 50–99 were checked by one Opus agent (42 fixed) and reviewed by a second, independent Opus agent (5 more wording fixes, no answer changed).
- **Swedish general questions:** 100 everyday questions (e.g. "Varför flyter en båt av stål?"); 60 written first, 40 added later and reviewed independently (all passed). Three of the first 60 were reworded for grammar.

## 3. Runs

- **Engine:** vLLM 0.34 (`scripts/run_vllm.py`), same prompts, scoring and output format as the plain transformers version (`scripts/run_experiment.py`), which was too slow for 8,192-token answers.
- **Settings in the report:** 100 items per test, up to 8,192 new tokens; greedy decoding (`results/n100_greedy_8k`) and the article's sampling settings, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, seed 0 (`results/n100_paper_8k_s0`).
- **Checks not in the report:** two more sampling seeds gave similar numbers (English math finished M0 7–10 vs M1 29–34; Swedish math 41–43 vs 77; Swedish general 63–68 vs 99–100). A 16,384-token run was started to see whether M1 finishes more long English problems.

## 4. Results (100 items per test; finished / correct or sensible)

| Test | Greedy M0 | Greedy M1 | Sampling M0 | Sampling M1 |
|---|---|---|---|---|
| English math | 0 / 0 | 14 / 14 | 10 / 10 | 34 / 31 |
| Swedish math | 2 / 2 | 35 / 25 | 41 / 34 | 77 / 52 |
| Swedish general questions | 7 / 3 | 32 / 15 | 63 / 34 | 99 / 44 |

- Repetition in the thinking (sampling): M0 25–42%, M1 0–8%. Greedy: M0 83–85%, M1 50–72%.
- M0's unfinished answers are loops (median 45% repetition with sampling). Most of M1's unfinished English math answers are not loops (44 of 66 repeat less than 10%): it is still reasoning at the limit.
- Among finished answers, M1 is not more accurate (Swedish math 52/77 vs 34/41; general 44/99 vs 34/63): the gain comes from finishing.
- Both models finish Swedish math more often than English math; not yet explained.
- Example used in the report: Swedish math item 44 (sampling): M0 loops on "Wait, is there any other interpretation?", M1 answers 3 (correct).

## 5. Judging

- A first A/B judge (two answers side by side) was discarded: it marked 8 "no answer" placeholders as sensible.
- Replaced by blind single-answer rating (`scripts/make_blind_singles.py`): only finished answers, M0 and M1 shuffled together without labels, rated by Claude Opus 5.5; an unfinished answer counts as not sensible. Greedy: M0 3, M1 15 sensible; sampling: M0 34, M1 44.

## 6. Decisions and review

- Report and repository use two settings only (greedy and the article's sampling, seed 0); the two serve as a check that the effect is not caused by one decoding setting. The focus is the qualitative M0 vs M1 difference and transfer across languages and tasks.
- An independent Opus reviewer recomputed every number in the report from the raw data (all correct) and flagged overclaiming. Fixes applied: "finished" as the measure; transfer shown for the same model only (it must be re-checked on a model trained further on Swedish); the fixed loops are in the model's English thinking; the gain comes from finishing, not from more accurate answers.
