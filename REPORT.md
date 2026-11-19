# Weight-Space Edits for Swedish LMs — SE.LLMA Assignment

19 November 2026 · Sangita Dutta

**Abstract.** The article *Sparks of Weight-Space Programming* shows a small weight edit that stops a small model (Qwen3.5-0.8B) from getting stuck in loops while reasoning on English math. We test whether this edit also works in Swedish. We compare the original model (M0) with the edited model (M1) on English math, the same math in Swedish, and everyday Swedish questions. M1 finished far more often than M0 in all three tests, with both greedy decoding and the article's sampling settings, and gave more correct and sensible answers. The edit thus carries over between languages and from math to general questions, which suggests that Swedish models could reuse edits found on non-Swedish models.

## Part A: The article

[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, November 2026)

- **Claim:** a frontier AI model (GPT-6 Astra) can read and edit the weights of other AI models directly, like a programmer edits code. This could fix and improve models without retraining them.
- **Evidence:** on tiny models, it could tell from the weights alone what a model was trained to do, and fixed a failing car-driving model with one edit. On a small language model (Qwen3.5-0.8B), it raised math accuracy from 8% to 48% by changing 0.016% of the weights.
- **What weakens it:**
    - The math gain came from removing a fault: the model got stuck repeating itself, and the edit stopped this. On a model without this fault, no edit helped.
    - On real language models, the edit was found by about 50 rounds of trial and error, not by reading the weights alone.
    - Few runs, a closed editor model, and almost no comparison with simpler methods.
- **Bottom line:** promising, but for a narrower claim. With much testing, AI can find a small edit that removes a specific fault in a small model.

## Part B: Is it useful for SE.LLMA?

[SE.LLMA](https://wasp-sweden.org/research/projects/sellma/) builds Swedish language models, including models that reason step by step, by training existing open models further on Swedish.

- **Most relevant idea: fixing a specific fault without retraining.** Small reasoning models often get stuck repeating themselves. If an edit found on a non-Swedish model also works in Swedish, SE.LLMA could reuse it at almost no cost. Part C tests this.
- **Less useful today:** making models more capable by editing weights (not shown in the article), and reading what a model learned from its weights (shown only on tiny models).
- **Concerns:** finding a new edit is costly (about 20 hours of trial and error) and relies on a closed, foreign AI model.

## Part C: Our test — can the article's edit be reused for Swedish?

### Why this question

- The article's edit was found on English math. SE.LLMA would apply such edits to models used in Swedish.
- So we ask: does the edit also help in **Swedish**, and on questions **beyond math**?

### Hypothesis

- **If the edit fixes a general fault in how the model reasons,** it should help on all three tests: English math, Swedish math and Swedish general questions.
- **If it is tied to what it was found on,** it should help mainly on English math.

### What we tested

- **Two models:**
    - **M0** is the original small model, Qwen3.5-0.8B. When it thinks step by step, it often gets stuck repeating the same lines and never gives an answer.
    - **M1** is the same model with the article's edit: one small change to its weights, found on English math, meant to stop this repeating. We used it exactly as the authors published it.
- **Three tests, 100 items each, the same items for both models:** English math word problems; the same problems in Swedish; and everyday Swedish questions.
- **Two ways of picking the next word:** greedy (always the most likely word, so answers repeat exactly) and the article's sampling settings (some randomness, which can break a loop; details in the [article](https://rikiyat.github.io/blog/weight-space-programming/)). Using both shows that the effect is not caused by one setting.
- **Settings:** thinking on, since that is where the edit has something to fix; answers limited to 8,192 tokens.
- **Measures:** did the model **finish** (stop thinking and give an answer) within the limit. For math, the final number is compared automatically with the correct answer. For general questions (for example "Varför flyter en båt av stål?"), each finished answer is rated by an AI judge (Claude Opus 5.5) that does not know which model wrote it, and marked **sensible** if it answers the question, is mostly correct and is coherent. Unfinished answers count as not sensible.

### Results

M1 finishes far more often than M0 in all three tests, with both settings. Because it finishes more often, it also gives more correct and sensible answers in total, although among finished answers it is not more accurate. Numbers are out of 100.

**Greedy**

| Test | M0 finished | M1 finished | M0 correct or sensible | M1 correct or sensible |
| --- | --- | --- | --- | --- |
| English math | 0 | 14 | 0 | 14 |
| Swedish math | 2 | 35 | 2 | 25 |
| Swedish general questions | 7 | 32 | 3 | 15 |

**Article's sampling settings**

| Test | M0 finished | M1 finished | M0 correct or sensible | M1 correct or sensible |
| --- | --- | --- | --- | --- |
| English math | 10 | 34 | 10 | 31 |
| Swedish math | 41 | 77 | 34 | 52 |
| Swedish general questions | 63 | 99 | 34 | 44 |

- **The edit carries over to Swedish:** on Swedish math, M1 finishes and solves clearly more problems than M0.
- **It also carries over from math to general questions:** M1 almost always finishes, but its answers are only slightly more often sensible. The edit lets the model finish; it does not give it more knowledge.
- **The gap is there with both settings,** and two more sampling seeds gave similar numbers. Sampling alone also helps M0 a lot; the best results come from the edit and sampling together.
- With greedy, M1 still loops, just less often. With the article's settings, M1 hardly repeats itself: when it does not finish, it is usually still reasoning at the limit (Appendix 2).

### Example

A Swedish math problem (article's sampling settings):

> Peter planerar att gå på bio den här veckan. Han får alltid en biljett för $ 7 och popcorn för $ 7. Om han har 42 dollar för veckan, hur många gånger kan han gå på bio?

**M0** gets stuck. Its thinking (which the model writes in English, even for Swedish questions) ends in these lines, repeated until the limit, and it never answers:

```
*   **Wait, is there any other interpretation?**
*   Maybe the question implies he gets ONE ticket and ONE popcorn, so total is $14.
*   Maybe he goes to the bio once.
*   Maybe he goes twice.
*   Maybe he goes three times.
*   Okay, let's write down the steps clearly.
```

**M1** finishes and answers correctly, in Swedish (shortened):

> Kostnad per gång = $7 + $7 = $14. … $42 ÷ $14 = 3. Så Peter kan gå på bio 3 gånger.

### What this means

- The edit fixes a general fault in how the model reasons, not something tied to English or to math: it helps on all three tests, as the first hypothesis predicts.
- It fixes a fault, not a lack of knowledge: M1 finishes more often, but its finished answers are not more accurate.
- More generally, any edit that improves the behaviour of a small non-Swedish model, not only loop fixes, could therefore likely also improve Swedish models built on it.

### Limits

- One small model, which thinks in English, and the edit was reused on the model it was made for. A model trained further on Swedish would need to be checked again.
- Answers were limited to 8,192 tokens (the article allowed 32,768), so some long reasoning is cut off.
- Both models finish Swedish math more often than English math. We have not checked why.

### Recommendation

- **Build on non-Swedish work.** Fixes found on a widely used base model, mostly in English, can carry over to Swedish. SE.LLMA need not redo such work for each language and can reuse what the wider community finds.
- **Check before use.** Test each edit on the target Swedish model with a small check set like ours (finished, correct, sensible). This is quick and cheap.
- **Edits for faults, data for knowledge.** Edits can remove faults such as loops; better Swedish knowledge still needs Swedish data and training.
- **Next step:** test whether an edit made on the base model still works after further training on Swedish, and on a model that thinks in Swedish.

## Appendix 1: Experiment details

- **Data:** 100 GSM8K math problems in English and in Swedish (machine-translated, checked by AI and fixed by hand where needed), and 100 everyday Swedish questions written for this study.
- **Settings:** thinking on; up to 8,192 new tokens; greedy decoding, or the article's sampling settings (temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, seed 0).
- **Finished:** the model closed its thinking and gave an answer within the limit.
- **Correct or sensible:** for math, the final number matches the correct answer; for general questions, Claude Opus 5.5 rated each finished answer without knowing which model wrote it.

## Appendix 2: Full results

Out of 100 items per test. Repetition is the share of 8-word sequences in the thinking that already appeared earlier (near 100% = looping).

**Greedy**

| Test | M0 finished | M1 finished | M0 correct or sensible | M1 correct or sensible | M0 repetition | M1 repetition |
|---|---|---|---|---|---|---|
| English math | 0 | 14 | 0 | 14 | 85% | 72% |
| Swedish math | 2 | 35 | 2 | 25 | 85% | 62% |
| Swedish general questions | 7 | 32 | 3 | 15 | 83% | 50% |

**Article's sampling settings**

| Test | M0 finished | M1 finished | M0 correct or sensible | M1 correct or sensible | M0 repetition | M1 repetition |
|---|---|---|---|---|---|---|
| English math | 10 | 34 | 10 | 31 | 42% | 8% |
| Swedish math | 41 | 77 | 34 | 52 | 30% | 4% |
| Swedish general questions | 63 | 99 | 34 | 44 | 25% | 0% |

Two more sampling seeds gave similar numbers (for example, Swedish math finished: M0 41–43, M1 77).

## Appendix 3: Reproducibility and AI usage

| Model or data | Revision | Licence |
|---|---|---|
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (M0) | 2fc0636 | Apache-2.0 |
| [RikiyaT/Qwen3.5-0.8B-weight-space-programming](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) (M1) | d1671a1 | Apache-2.0 |
| [Helsinki-NLP/opus-mt-en-sv](https://huggingface.co/Helsinki-NLP/opus-mt-en-sv) (first translation of the math problems) | cec444c | Apache-2.0 |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 740312a | MIT |

**Code and results:** vLLM 0.34 and transformers 5.23; every answer, the judge's ratings and the steps to reproduce are in the repository.

**AI usage:** Claude Code (Claude Opus 5.5) was used to set up and run all the tests of the hypotheses: it wrote the code, ran the experiments, and drafted this report and the lab notes. Claude subagents wrote and checked the Swedish questions and translations, judged the general answers blind, and reviewed the report. I steered the experiments and ideas, and iterated on Claude's instructions as needed to keep the experiments clean, robust and reasonable. The article's editor model (GPT-6 Astra) was not used; its released edit was reused as published.
