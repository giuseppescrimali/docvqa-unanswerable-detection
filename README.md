# Unanswerable Question Detection in Document VQA (Vision LLMs)

Can small Vision-Language Models tell when a question about a document **can't** be answered — by actually looking at the document, rather than guessing from the text alone?

This project builds a synthetic corruption benchmark on top of DocVQA to test exactly that, evaluates three small (2B–4B) Vision LLMs on binary answerable/unanswerable classification, and compares three prompt-only mitigation strategies across single-page and multi-page settings.

**Full write-up:** [`Scrimali_Giuseppe_SpiegazioneSoluzione.pdf`](./Scrimali_Giuseppe_SpiegazioneSoluzione.pdf)

## Why this problem

Most DocVQA benchmarks assume every question is answerable. In practice, a document QA system needs to know when to say "I can't find that" instead of hallucinating an answer — and it needs to make that call by inspecting the *visual* content of the document, not just by pattern-matching the question text. This project isolates that specific capability.

## Approach

1. **Synthetic corruption benchmark.** Starting from answerable questions in SP-DocVQA and MP-DocVQA, three rule-based corruption types make a question unanswerable while keeping it grammatically plausible:
   - **Entity** — a name/date/organization is swapped via NER (spaCy) for an out-of-domain value → tests text reading & matching.
   - **Element** — a page-element reference (e.g. *table → figure*) is swapped for a semantically distant one → tests visual structure understanding.
   - **Layout** — a spatial reference (e.g. *top-left → bottom-right*) is flipped → tests spatial localization.

2. **Quality control via LLM-as-a-judge.** Every corrupted question is checked by Llama-3.3-70B for grammaticality and for genuinely requiring the document (not just common sense) to detect. Failed samples are rewritten once, then dropped if still failing — keeping the final set balanced 50/50 (198 samples: 99 corrupted, 99 clean).

3. **Model evaluation.** Three instruction-following Vision LLMs of comparable size (2B–4B) — Qwen2-VL-2B, Gemma-3-4B, Phi-3.5-Vision — are prompted as binary classifiers and scored on F1 (unanswerable class) and clean accuracy, run in isolated sessions due to incompatible `transformers` version requirements.

4. **Mitigation strategies**, applied purely via prompting (no fine-tuning): **few-shot** examples, explicit **knowledge injection** (definitions of the three corruption types), and **chain-of-thought** reasoning.

5. **Multi-page extension.** MP-DocVQA lets the model see a window of *W* pages around the answer page, testing whether extra visual context helps or hurts.

## Key findings

- **The smallest model won.** Qwen2-VL-2B (2B params) was the only model with strong F1 across *all three* corruption types (F1 = 0.58 baseline). Gemma-3-4B and Phi-3.5-Vision, despite being larger, showed a strong bias toward answering "answerable" regardless of document content (recall 0.35 and 0.22 respectively).
- **No mitigation is universally effective.** Knowledge injection helped Phi (F1 0.34 → 0.58) but not Gemma, which stayed stuck on its answerable bias across every strategy. The same strategies pushed Qwen toward the *opposite* bias, collapsing clean accuracy to below 0.21.
- **Few-shot was the one robust strategy**, improving F1 across all three models without collapsing clean accuracy — and produced the study's best result (F1 = 0.70 on Qwen).
- **Chain-of-thought made things worse** for most models — a sign that explicit multi-step reasoning needs more parametric capacity than a 2–4B model has.
- **Extra page context (multi-page, W=2) has no uniform effect**: it helped Gemma detect structural corruptions but pushed Qwen and Phi further toward "answerable," diluting attention on the relevant page.

## Repo contents

| File | Description |
|---|---|
| `pipeline_singlepage.py` | Full pipeline (corruption → LLM-judge filtering → inference → mitigation strategies) for SP-DocVQA |
| `pipeline_multipage.py` | Multi-page extension (MP-DocVQA), including the page-windowing logic (`get_window_pages`) |
| `Scrimali_Giuseppe_SpiegazioneSoluzione.pdf` | Full report with methodology, all figures/heatmaps, and detailed per-model analysis |

## Tech stack

`transformers` · Qwen2-VL / Gemma-3 / Phi-3.5-Vision · spaCy (NER) · Llama-3.3-70B via Groq (LLM-as-a-judge) · Google Colab

## Data

[DocVQA / SP-DocVQA](https://www.docvqa.org/docvqa) and [MP-DocVQA](https://rrc.cvc.uab.es/?com=contestant) (Mathew et al., 2021; Tito et al., 2023).

---

*Part of my AI/ML portfolio — see [profile README](https://github.com/giuseppescrimali) for other projects.*
