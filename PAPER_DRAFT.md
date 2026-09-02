# Eng-PidginEdu: A Glossary-Augmented Benchmark for English–Nigerian Pidgin Machine Translation

**Status: draft.** This is a first full draft assembled from the project's
complete experimental record (`BENCHMARK_REPORT.md` in this repository),
restructured into standard paper form. It is not yet formatted for a
specific venue's template (no LaTeX, no page limit applied), citations
are limited to works directly verified during this project (see the
note at the start of *References*), and the author list, affiliations,
and acknowledgments are placeholders. Treat every number in this draft
as traceable to `BENCHMARK_REPORT.md` -- if a number here and there
disagree, the report is the source of truth.

## Abstract

We present Eng-PidginEdu, a benchmark for English-to-Nigerian Pidgin
machine translation built around a novel form of supervision:
glossary-augmented targets, in which academic terminology in the
translation is followed by an inline parenthetical gloss in Pidgin
(e.g. *"di judiciary (di part of government wey de interpret and use
law)"*). We release a 26,232-sentence parallel corpus spanning eight
Nigerian secondary-school subjects, a 13,634-entry English-Pidgin
terminology glossary, and the verified preprocessing pipeline that
merges them. We introduce **glossary-accuracy**, a metric that scores
whether a model reproduces the correct gloss for each in-context
technical term, independent of general translation quality. We
evaluate 14 multilingual sequence-to-sequence models spanning five
architecture families under three regimes -- zero-shot, full
fine-tuning, and low-rank adaptation (LoRA) -- against four metrics
(BLEU, chrF++, AfriCOMET, and glossary-accuracy), and report every
infrastructure failure, hyperparameter correction, and methodological
course-correction encountered along the way. Our central findings are:
(1) fine-tuning closes an architecture-independent gap to a common
performance ceiling, with the models weakest zero-shot gaining the
most; (2) LoRA matches full fine-tuning almost exactly for
BART-derived architectures (95-103% of full fine-tuning's
glossary-accuracy) but recovers only 20-58% of it for smaller
T5-family checkpoints, a gap not explained by parameter count alone;
and (3) the model that scores highest by every automated metric is not
the one we release as the project's flagship, a deliberate,
disclosed departure from a metrics-only selection process that we
document as a methodological case study in its own right. All code,
data, trained checkpoints, and the full experimental log are public.

## 1. Introduction

Nigerian Pidgin (ISO 639-3: `pcm`) is spoken by an estimated tens of
millions of people as a first or second language and functions as a
lingua franca across Nigeria and parts of West Africa. Despite this
reach, it remains low-resource in the machine translation literature:
none of the major multilingual MT systems we evaluate in this work
(NLLB-200, mBART-50, M2M-100, MADLAD-400, SeamlessM4T) include Pidgin
in their target-language inventory at all, and existing African-NLP
checkpoints (AfriTeVa, Cheetah, Toucan) are pretrained on a large set
of African languages that does not privilege Pidgin specifically.

This work targets a specific, understudied use case: **education**.
Nigerian secondary-school instruction is conducted in English, but
learners are frequently more comfortable receiving explanation in
Pidgin, particularly for technical vocabulary drawn from Computer
Science, Government, Business Studies, and other subjects with heavy
domain-specific terminology. Plain translation is not sufficient for
this use case: replacing an English technical term with its closest
Pidgin sentence-level paraphrase can lose the term itself, which a
learner still needs to recognize in exams, textbooks, and further
study. We therefore target a different output structure: **the term
is retained, and its meaning is glossed inline**, in the style of a
textbook footnote compressed into the sentence.

### 1.1 Contributions

1. **Eng-PidginEdu**, a 26,232-sentence English-Pidgin parallel corpus
   across eight academic subjects, and a 13,634-entry terminology
   glossary, both released under CC BY 4.0.
2. **A verified glossary-augmentation pipeline** (`glossary_augment.py`)
   that merges the corpus and glossary into inline-glossed training
   targets, confirmed to reproduce the released augmented dataset
   exactly, row for row.
3. **Glossary-accuracy**, a novel metric scoring whether a model
   reproduces the correct inline gloss for each technical term
   present in a reference translation, reported alongside BLEU,
   chrF++, and AfriCOMET.
4. **A 14-model, 3-regime benchmark** (zero-shot, full fine-tuning,
   LoRA/PEFT) spanning five architecture families and a parameter
   range from 229M to 3B, with every training run's hyperparameters,
   failures, and corrections documented.
5. **PidginEdu-LLM**, a released flagship checkpoint, and an explicit,
   documented account of why its selection departed from the
   benchmark's own automated-metric ranking.

### 1.2 Research questions

- **RQ1.** How well do existing multilingual models perform on
  English-to-Pidgin translation without any adaptation?
- **RQ2.** How much does fine-tuning on Eng-PidginEdu improve
  translation, and does that improvement depend on model family?
- **RQ3.** Which pretrained model is most adaptable to Nigerian Pidgin?
- **RQ4.** Can parameter-efficient adaptation (LoRA) match full
  fine-tuning?
- **RQ5.** How does model scale relate to translation performance in
  this setting?
- **RQ6.** Does the resulting flagship model hold up against the rest
  of the benchmarked roster?

## 2. Related Work

**Multilingual MT backbones.** We evaluate checkpoints from mT5 (Xue
et al., 2021), M2M-100 (Fan et al., 2021), NLLB-200 (NLLB Team et al.,
2022), mBART-50 (Tang et al., 2020; Liu et al., 2020), MADLAD-400
(Kudugunta et al., 2023), SeamlessM4T v2 (Barrault et al., 2023), and
T5 v1.1 (Raffel et al., 2020), alongside two checkpoints pretrained
specifically for African languages: AfriTeVa (Jude Ogundepo et al.,
2022) and Cheetah/Toucan (Adebara et al., 2024), the latter evaluated
directly on Toucan's own held-out benchmark, AfroLingu-MT, in its
original paper.

**Parameter-efficient fine-tuning.** We use LoRA (Hu et al., 2021) as
implemented in the HuggingFace `peft` library, which freezes the base
model and injects trainable low-rank update matrices into selected
linear layers.

**Evaluation.** We report BLEU (Papineni et al., 2002) and chrF++
(Popović, 2015) via `sacrebleu`, and AfriCOMET-STL (Wang et al., 2023),
a COMET-style neural metric fine-tuned for African-language MT
evaluation that explicitly includes Pidgin in its training language
set -- notable because none of the multilingual MT systems above do.

*[Placeholder for the author: this section should be expanded with
prior work specifically on Nigerian Pidgin MT and on terminology-
constrained / glossary-aware MT more broadly. One candidate system,
Pidgin-UNMT, was investigated during this project (see
`BENCHMARK_REPORT.md` §9.1) but could not be run as a baseline -- its
linked pretrained checkpoint was no longer retrievable. A literature
search beyond what was directly encountered during this project's own
experimentation has not been performed and should not be assumed done.]*

## 3. The Eng-PidginEdu Dataset

### 3.1 Corpus construction

The corpus comprises 26,232 English-Pidgin sentence pairs drawn from
Nigerian secondary-school educational materials across eight subjects
(Computer Science, Business Studies, Government, Social Studies,
Biology, English Language, History, Civic Education). Construction
proceeded in five stages: document extraction from source PDFs/DOCX
files (`PyPDF2`, `python-docx`); sentence segmentation (NLTK Punkt);
normalization and cleaning (spaCy, textacy, regex filtering); draft
translation generation (`Davlan/mt5-small-en-pcm`); and human
validation and correction by native Nigerian Pidgin speakers. The
corpus is randomly split 80/10/10 (train/dev/test) under a fixed seed
for reproducibility.

### 3.2 Glossary augmentation

A separate 13,634-entry glossary (`Technical_terms`, `Literal
meaning`, `pidgin meaning`) provides Pidgin explanations for
education-domain English vocabulary. `glossary_augment.py` merges this
glossary into the corpus's Pidgin targets, producing
`pcm_augmented`: for each sentence, matched glossary terms
(case-insensitive, simple plural matching) are annotated inline with
their Pidgin meaning in parentheses, subject to filters that exclude
overly common terms (document frequency > 0.2% of sentences), overly
long glosses (> 12 words), and glosses that merely restate the term.
Where a term has multiple candidate glossary entries, **the first
occurrence in the glossary file decides the term's fate outright** --
later duplicate entries are never consulted, even if the first
entry's gloss was itself rejected by the filters above. This behavior
was not documented anywhere and had to be recovered by testing
candidate implementations against the already-published augmented
corpus until one reproduced it exactly (see `BENCHMARK_REPORT.md` §12
item 4 for the full account, including an unrelated pandas
NA-string-handling interaction that also had to be diagnosed before an
exact match was achieved).

The resulting augmentation covers 61.4% of sentences (16,114/26,232),
inserting 28,243 glosses across 2,473 distinct terms.

**Example.**
```
EN:  The ability for educators and students to adopt the materials
     for free.
PCM: Teachers and students fit get the materials for free.
AUG: Teachers and students fit get the materials (tins wey dem use
     for teaching) for free.
```

## 4. The Glossary-Accuracy Metric

Standard MT metrics (BLEU, chrF++) are computed over full sentences
and are insensitive to whether a specific technical term was correctly
glossed -- a model can score well overall while systematically failing
to produce any glosses at all (we observe exactly this failure mode in
§6.4). We therefore score glossary fidelity as an independent metric.

For each reference sentence containing one or more inline glosses, we
extract the glossed term(s) and their reference gloss text. We then
check whether the model's *hypothesis* translation mentions the term
(`term_mention_rate`) and, separately, whether it produces a
parenthetical gloss for that term whose content sufficiently matches
the reference gloss under a chrF-based similarity threshold
(threshold: 50). From this we derive:

- **Gloss presence rate**: fraction of expected glosses that appear
  (in any form) in the hypothesis.
- **Gloss accuracy**: fraction of expected glosses that appear *and*
  match the reference gloss content above threshold.
- **Gloss precision / over-glossing rate**: how much of what the model
  glosses was actually expected, penalizing spurious glosses.
- **Gloss F1**: harmonic mean of gloss accuracy (recall-oriented) and
  gloss precision.

BLEU and chrF++ are additionally reported against two reference
conditions: the **clean** reference (glosses stripped from both
hypothesis and reference, for comparability with standard MT
literature) and the **augmented** reference (scored as-is, only
comparable within this benchmark).

## 5. Models and Training Regimes

### 5.1 Roster

Fourteen checkpoints spanning 229M to 3B parameters and five
architecture families: T5-style encoder-decoders (AfriTeVa,
AfriTeVa-v2-Large, mT5-base, mT5-Large, AfriMT5, Cheetah, Toucan,
MADLAD-400-3B, T5-v1.1-XL) and BART-style encoder-decoders (M2M-100
418M, M2M-100 1.2B, NLLB-200-distilled-600M, mBART-50,
SeamlessM4T-v2-Large). Full checkpoint identifiers and parameter
counts are in `BENCHMARK_REPORT.md` §6.

None of the five multilingual (non-Africa-specific) checkpoints have
Nigerian Pidgin in their target-language inventory. For the four that
require an explicit language token (NLLB, mBART, M2M-100, SeamlessM4T),
we substitute the closest available proxy -- Tok Pisin (`tpi_Latn`) for
NLLB, the only English-lexified creole in its inventory -- or fall back
to the English token where no creole option exists at all (mBART,
M2M-100, SeamlessM4T). This is a genuine confound relative to the
T5-family models, which condition on a plain-text instruction prefix
instead and face no equivalent identity conflation; we flag it
explicitly rather than treat it as a detail (`BENCHMARK_REPORT.md`
§3.12, §9.2).

### 5.2 Training regimes

- **Zero-shot**: the original pretrained checkpoint, no Pidgin
  exposure, greedy decoding.
- **Full fine-tuning**: all parameters updated, fp32, Adafactor for
  the 1B+ tier (memory) and AdamW otherwise, 5 epochs, early stopping
  on validation chrF, beam-5 decoding for final scoring. Two
  checkpoints (MADLAD-400-3B, T5-v1.1-XL) could not be fully
  fine-tuned on the available hardware (2×32GB V100, no
  sharded-optimizer support in this pipeline) -- their genuinely
  untied input/output embeddings double the effective memory cost of
  their embedding table on top of full fp32 optimizer state, exceeding
  available memory before any activation memory is counted.
- **LoRA**: base model frozen, rank-16 low-rank adapters on the
  attention projections (and, after a correction described in §6.3,
  the feed-forward projections), learning rate 1e-3, AdamW. LoRA
  unblocks both models excluded from full fine-tuning, since the
  memory bottleneck there is optimizer state, not model capacity.

Full hyperparameters for every model under every regime, and the
wall-clock/compute cost of each run, are in `BENCHMARK_REPORT.md` §6,
§7.4, and §11.1.

## 6. Results

### 6.1 RQ1: Zero-shot capability

Zero-shot BLEU ranges from 32.71 (SeamlessM4T) down to 0.00 (Cheetah)
across the 14 models. SeamlessM4T and both M2M-100 variants retain
substantial capability without any Pidgin exposure; most T5-family
checkpoints are close to non-functional for this language pair
zero-shot. Full per-model scores in `BENCHMARK_REPORT.md` §9.7.

### 6.2 RQ2/RQ3: Fine-tuning gain and adaptability

Across the 12 models with both a zero-shot and full-fine-tuned score,
gains range from +34.85 to +66.16 BLEU, and **gain is inversely
related to zero-shot starting point**: all 12 converge to a similar
final BLEU band (57.7-68.5) regardless of where they started. We
consider this the benchmark's single most consequential finding for
motivating the dataset's value: fine-tuning on Eng-PidginEdu closes an
architecture-independent gap to a common ceiling, rather than
primarily amplifying models that were already strong.

"Most adaptable" admits two readings that this benchmark can now
distinguish empirically. By raw fine-tuning gain, Cheetah improves the
most (+66.16 BLEU) -- but this measure is confounded with starting
point, since it correlates almost entirely with how weak a model was
zero-shot. A cleaner adaptability signal is **LoRA retention**: how
much of a model's own full-fine-tuning quality is reachable through a
lightweight update alone (§6.3). By that measure, SeamlessM4T,
NLLB, mBART-50, and both M2M-100 variants are the most adaptable
models in the roster, retaining 95-103% of their full-fine-tuning
glossary-accuracy under LoRA.

### 6.3 RQ4: Full fine-tuning vs. LoRA

LoRA does not cost translation fluency broadly: BLEU/chrF++ stay
within a narrow band across the roster regardless of adaptation
method. Glossary-accuracy tells a different story. An initial LoRA
configuration restricted to attention projections (`q`, `v` -- the
original LoRA paper's default) left glossary-accuracy severely
degraded for several models despite normal-looking loss curves and
BLEU scores -- most dramatically, AfriTeVa's glossary accuracy
collapsed from 41.71% (full fine-tuning) to 0.04% under attention-only
LoRA, while producing fluent, on-topic translations that simply never
inserted the expected glosses. We diagnose this as a content-injection
problem specific to attention-only adaptation: attention layers
reweight what a model attends to, but new lexical content -- which a
specific, correct gloss for a specific term requires -- is generated in
the feed-forward layers, which were left frozen. Extending
`target_modules` to include the feed-forward projections
(`wi`,`wo` for T5-family checkpoints; `fc1`,`fc2` for BART-style ones)
recovered most of this gap.

The corrected, final picture across all 14 models is a clean
architecture split, not noise:

| Architecture group | LoRA / full-FT glossary-accuracy retention |
|---|---|
| BART-style (Seamless, NLLB, mBART-50, both M2M-100 variants) | 95-103% |
| 1B+ T5-family (Toucan, Cheetah, AfriTeVa-v2-Large, mT5-Large) | 74-83% |
| Base-tier T5-family (AfriTeVa, AfriMT5, mT5-base) | 24-58% |

We do not have a confirmed mechanistic account of the third band --
a plausible hypothesis (smaller pretrained models have less latent
capacity for a low-rank update to redirect, so a task closer to new
knowledge injection than to fluency-steering needs more of it than a
rank-16 adapter provides) is offered but not verified against a direct
ablation, and is flagged as an open question rather than a settled
finding.

### 6.4 RQ5: Scale vs. performance

Across the 229M-3B range actually trained in this benchmark (no
checkpoint above 3B was trainable on the available hardware), we find
a **sharp threshold effect followed by a plateau**, not a continuous
scaling trend. The smallest model (AfriTeVa, 229M) is a clear outlier,
well below every other model on every metric; from 418M upward, all
four metrics cluster tightly regardless of further scale increases,
and the two largest models tested (2.85B/3B, LoRA-only) do not
outperform the 600M-1.4B tier on any metric. Pearson correlation
between parameter count and performance across the 12 full-fine-tuned
models drops from 0.62-0.72 (including the 229M outlier) to 0.49-0.75
once it is excluded -- most of the apparent scale relationship is a
single small-model penalty, not a continuous trend among the
mid-to-large checkpoints. Architecture and adaptation method explain
more of the variance in this dataset than raw parameter count does
once a model clears roughly 400-600M parameters.

### 6.5 RQ6 and flagship selection

Ranked purely by automated metrics, `mt5_large` leads 3 of 4 (GlossF1
78.51, AfriCOMET 71.94, chrF++ 80.59) among the 12 fully fine-tuned
models, confirmed on both the test split and, after we identified and
corrected a test-set-only selection procedure as a multiple-comparison
risk, on an independent validation-split leaderboard as well (full
account in `BENCHMARK_REPORT.md` §9.5).

**We do not release `mt5_large` as the project's flagship.** After the
metrics-only process concluded, the project author reviewed generated
translations from `mt5_large` and from `toucan` (73.96 GlossF1 --
seventh of twelve on the same ranking) directly, and judged `toucan`'s
Pidgin phrasing more natural and fluent despite its lower scores on
every automated metric used here. **PidginEdu-LLM is `toucan`**, and
we state this override plainly rather than let the metrics section
stand as if it had produced this outcome: it is a single-reviewer
qualitative judgment made after, and in spite of, a process built
specifically to avoid exactly this kind of ad hoc override, not a new
evaluation protocol with a defined sample, blind raters, or an
inter-rater agreement figure. We surface it as a case study for the
field rather than smoothing it over: automated MT metrics and human
perception of output quality are known to diverge, and this benchmark
produced a concrete instance of that divergence large enough to change
which checkpoint gets called the flagship, on a dataset and metric
suite built specifically to measure the thing (terminology fidelity)
that the losing-by-the-numbers model was chosen for.

## 7. Limitations

- **No formal human evaluation.** §6.5's flagship decision is
  explicitly not a substitute for one; a rigorous resolution of the
  automated-vs-human divergence it surfaces would need a defined
  sample, multiple blind raters, and an agreement statistic, none of
  which this benchmark collected.
- **Proxy language-token confound** (§5.1) affects five of the
  fourteen models and is not present for the T5-family checkpoints,
  making family-level comparisons not fully apples-to-apples.
- **Education-domain corpus only**; results may not transfer to
  conversational or general-domain Pidgin.
- **Scale range capped at ~3B** by available hardware; RQ5's findings
  should not be extrapolated to larger models without further
  evidence.
- **The base-tier T5 LoRA gap (§6.3) is unresolved**, not just
  unexplained -- we have not run the follow-up ablations (higher rank,
  more epochs) that would distinguish our capacity hypothesis from
  alternative explanations.

## 8. Conclusion

Eng-PidginEdu contributes a glossary-augmented parallel corpus, a
metric designed specifically to measure terminology fidelity rather
than general fluency, and a 14-model benchmark spanning three
adaptation regimes that produces two findings with implications beyond
this specific language pair: fine-tuning gain is inversely related to
zero-shot capability across every architecture tested, suggesting
domain-specific supervision closes a shared gap rather than amplifying
existing strength; and parameter-efficient adaptation's effectiveness
for terminology-specific tasks depends on which layers are adapted and
on base-model scale in a way that plain translation-quality metrics do
not reveal. We release all code, data, and checkpoints, and we
document a flagship-selection decision that departed from our own
metrics in enough detail that a reader can agree or disagree with it
on the same evidence we had.

## References

*Only works whose titles/authors/venues were directly encountered and
confirmed during this project (via model cards, package documentation,
or citations already present in project artifacts) are listed here.
Full bibliographic details (exact page numbers, DOIs) have not been
independently verified and must be checked before submission -- treat
every entry below as needing a final citation-accuracy pass, not as
submission-ready.*

- Adebara, I. et al. (2024). Toucan: Many-to-Many Translation for 150
  African Language Pairs. *Findings of ACL 2024.*
- Fan, A. et al. (2021). Beyond English-Centric Multilingual Machine
  Translation. (M2M-100)
- Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language
  Models.
- Kudugunta, S. et al. (2023). MADLAD-400: A Multilingual And
  Document-Level Large Audited Dataset.
- Liu, Y. et al. (2020). Multilingual Denoising Pre-training for
  Neural Machine Translation. (mBART)
- NLLB Team et al. (2022). No Language Left Behind: Scaling
  Human-Centered Machine Translation.
- Papineni, K. et al. (2002). BLEU: a Method for Automatic Evaluation
  of Machine Translation.
- Popović, M. (2015). chrF: character n-gram F-score for automatic MT
  evaluation.
- Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning
  with a Unified Text-to-Text Transformer. (T5)
- Wang, J. et al. (2023). AfriCOMET: Bridging the Gap in Machine
  Translation Evaluation for African Languages. arXiv:2311.09828.
- Xue, L. et al. (2021). mT5: A Massively Multilingual Pre-trained
  Text-to-Text Transformer.

## Appendix A: Reproducibility

Full hyperparameters for every model under every regime, every
infrastructure failure encountered and its root cause and fix,
compute-cost tables, and step-by-step reproduction instructions are in
this repository's `BENCHMARK_REPORT.md` and `README.md`. Code, dataset
(CC BY 4.0), and trained checkpoints are public:

- Code and full experimental report: `github.com/shashacode/Eng-PidginEdu-Benchmark`
- Dataset: `huggingface.co/datasets/coderGit/Eng_PidginEdu`
- PidginEdu-LLM (flagship): `huggingface.co/coderGit/eng-pidginedu-toucan`
