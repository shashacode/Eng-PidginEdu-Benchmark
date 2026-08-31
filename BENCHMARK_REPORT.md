# Eng-PidginEdu Benchmark: Progress Report

**Scope.** This documents every model run in the English→Nigerian
Pidgin glossary-augmented MT benchmark: the training pipeline, the
infrastructure failures encountered and their root causes, the glossary-
accuracy metric design, and full results including a same-architecture
before/after comparison that isolates why several models initially
failed to learn terminology glossing (or, for `toucan`/`cheetah`, to
train at all) before being corrected. §13 answers this project's six
research questions directly, each pointing to its supporting evidence
elsewhere in the report.

**Status as of this report.** The original 10-model sweep completed
(afriteva, mt5, afrimt5, m2m100, nllb, mbart50, afriteva_v2_large,
mt5_large, toucan, cheetah -- §7), after which the roster was revised
for paradigm diversity (§9): madlad400-7b-mt, mT5-XL, Cheetah-3.7B, and
TranslateGemma-4B were dropped in favor of MADLAD-400 3B/7B, T5
v1.1-XL, SeamlessM4T v2 Large, and (replacing madlad-7b specifically
once it proved infeasible) M2M-100 1.2B. **12 of the eventual 14
models are now fully fine-tuned and scored** -- `seamless` and
`m2m100_1.2b` (§9.6) both landed without needing correction, unlike
`toucan`/`cheetah` before them. `madlad3b` and `t5-v1_1-xl` hit a
genuine memory ceiling (§9.3) and are deferred to the LoRA phase, not
abandoned. **`toucan` is finalized as PidginEdu-LLM** (§9.5) -- not
the automated-metric leader (that is `mt5_large`, which leads 3 of 4
metrics: GlossF1, AfriCOMET, chrF++), but selected by the project
author's own qualitative judgment of translation quality after
inspecting outputs from both, on the basis that `toucan`'s Pidgin
reads more naturally and fluently despite scoring lower on every
automated metric. This is an explicit, disclosed departure from the
metrics-only selection criterion this benchmark originally committed
to (§9.5/§12 item 8) -- documented as exactly that, not folded in as
if it were always the plan.

**AfriCOMET is now implemented** (§10) and has scored all 12 completed
models -- the evaluation framework's four metrics (BLEU, chrF++,
AfriCOMET, glossary-accuracy) are all in place. **The zero-shot sweep
is complete** (§9.7): all 14 models scored, producing the
fine-tuning-gain table that is arguably this benchmark's single most
paper-relevant finding (an inverse relationship between zero-shot
capability and fine-tuning gain, with every model converging to a
similar final BLEU band regardless of starting point).

**LoRA/PEFT fine-tuning (§11, RQ4) is complete.** All 14 models now
have a LoRA result. `madlad3b` and `t5-v1_1-xl` -- the two models
excluded from full fine-tuning by a genuine memory ceiling (§9.3) --
are confirmed unblocked by LoRA as intended. A real methodological bug
was found and fixed along the way (§11.2): LoRA's default
attention-only target modules (`q`,`v`) left the glossary-accuracy
metric badly degraded relative to full fine-tuning even where
translation quality looked fine, and adding the feed-forward
projections recovered most of the gap. The final RQ4 picture is a
clean architecture split: every BART-style model (`seamless`, `nllb`,
`mbart50`, `m2m100`, `m2m100_1.2b`) retained 95-103% of its full-FT
glossary score under LoRA, while the base-tier T5-family models
(`afriteva`, `afrimt5`, `mt5`) retained only 20-58% -- a genuine,
unresolved finding, not an artifact (§11.4). Remaining open items
(dataset release, terminology-annotation code, etc.) are in §12.

---

## 1. Hardware and software environment

| | |
|---|---|
| GPUs | 2× Tesla V100-SXM3-32GB (compute capability 7.0, Volta) |
| Driver | NVIDIA 580.159.03, CUDA 13.0 |
| OS / kernel | Ubuntu, Linux 6.8.0-124-generic (x86_64) |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu126 (see §3.1 -- this specific build was required) |
| Transformers | 4.57.6 (see note below -- drifted from 5.15.0 mid-project) |
| Accelerate | 1.14.0 (see §11.1 for a launcher-path bug found in this version) |
| PEFT | 0.20.0 (§11 -- LoRA/PEFT fine-tuning) |
| Datasets | 5.0.1 |
| evaluate | 0.4.6 |
| sacrebleu | 2.6.0 |
| sentencepiece | 0.2.2 |
| pandas | 3.0.5 |
| unbabel-comet | 2.2.7 (§10 -- AfriCOMET) |
| pytorch-lightning | 2.6.5 (comet dependency) |
| torchmetrics | 0.10.3 (comet dependency) |
| Precision | fp32 throughout (see §3.1 -- no bf16 on Volta; T5-family is unstable in fp16) |
| Parallelism | PyTorch DistributedDataParallel via `torchrun`, one process per GPU (§11.1: single-GPU LoRA runs also went through `torchrun --nproc_per_node=1` rather than plain `python`, to work around an `accelerate` launcher bug) |

**Transformers version note, found while packaging this project for
reproduction.** The earliest full-fine-tuning runs in this project used
transformers 5.15.0. Installing `peft`/`unbabel-comet` later in the
project (for LoRA and AfriCOMET, §10/§11) pulled in 4.57.6 via normal
dependency resolution, and every result from the AfriCOMET
implementation onward -- the zero-shot sweep, the entire LoRA phase,
and the §3.20 fix and re-verification -- actually ran under 4.57.6,
not 5.15.0 as originally documented here. Checked directly rather than
assumed whether this matters: the two behaviors this report's comments
attribute to "transformers 5.x" (the `dtype=` kwarg replacing
`torch_dtype=`, and `forced_bos_token_id` on `model.config` becoming a
hard error rather than a deprecation warning, §3.10) both check out
identically under 4.57.6 -- `dtype=` genuinely controls load precision,
confirmed by inspecting a loaded model's parameter dtype directly, and
the workaround this codebase already uses (`generation_config` instead
of `model.config`) is correct and necessary under both versions
regardless of whether the older one hard-errors or merely warns. No
behavioral drift found that would change any reported number.
`requirements.txt` pins 4.57.6, the version verified to still work end
to end as of this fix, not the original 5.15.0.

---

## 2. Dataset

**Source:** `Eng-PidginEdu_glossary_augmented.csv`, 26,232 English–Pidgin
sentence pairs across 8 subjects (Computer science 10,917; Business study
5,550; Government 4,357; Social studies 1,978; Biology 1,531; English
Language 1,440; History 456; Civic Education 3).

**Glossary augmentation.** 16,114 of 26,232 rows (61.4%) carry one or more
inline terminology glosses: an English term is followed by a parenthetical
Pidgin explanation, e.g.

> Di note say race and color no dey on di list of **acceptable (dem worthy
> of acceptance or satisfactory)** bfoqs

There are 2,473 unique glossed terms across the corpus, all single words.
This inline-gloss format is the basis of the glossary-accuracy metric
(§4).

**Splits.** train 20,986 / dev 2,623 / test 2,623 (80/10/10), partition
fixed by the original data preparation, recovered exactly by
`prepare_data.py` (see §3.3) -- 26,232/26,232 CSV rows matched, 0 unclaimed,
0 cross-split overlap.

| Split | Rows | Glossed rows | Total glossed terms |
|---|---|---|---|
| train | 20,986 | 12,858 | 22,571 |
| dev | 2,623 | 1,630 | 2,814 |
| test | 2,623 | 1,626 | 2,858 |

**Target column.** `pcm_augmented` (the glossed translation) is used as
the training target for every model in this benchmark, so all reported
models are the glossary-augmented condition. `pcm_clean` (gloss-free) is
retained in the splits for ablation and for scoring standard-comparable MT
metrics (§4.3).

---

## 3. Infrastructure work and bugs found

This section is included because every one of these was silent or
misleading in its initial symptom, and each shaped a modelling decision
worth stating in a methods section.

### 3.1 CUDA kernel mismatch (blocking, first failure)

**Symptom:** `torch.AcceleratorError: CUDA error: no kernel image is
available for execution on the device`, on the very first CUDA op.

**Cause:** the installed PyTorch build (`2.13.0+cu130`) was compiled for
compute capabilities `{7.5, 8.0, 8.6, 9.0, 10.0, 12.0}` and does not
include Volta (`sm_70`) kernels, which the V100 requires.

**Fix:** reinstalled `torch==2.13.0+cu126`, whose kernel set includes
`sm_70`. Verified directly (`torch.cuda.get_arch_list()` includes
`sm_70`; a test matmul on `cuda` succeeds) rather than assumed from the
version string.

### 3.2 NCCL/CUDA library mismatch (blocking, second failure)

**Symptom:** `RuntimeError: NCCL Error 1: unhandled cuda error`, only
under multi-GPU (DDP) launch.

**Cause:** the cu126 wheel was first installed with `--no-deps`, which
left the previously-installed cu13 NCCL library in place -- a version
mismatch between torch's expected NCCL and the one actually loaded.

**Fix:** reinstalled with dependencies included, pulling matching
`nvidia-nccl-cu12` and the rest of the cu12 stack.

### 3.3 Split files lacked glossary metadata

**Symptom:** not a crash -- a silent methodological gap. The original
`train/dev/test.json` carried only `{"en", "pcm"}`, where `"pcm"` was
already the glossary-augmented text. This is sufficient to train on, but
insufficient to score glossary accuracy, since scoring requires knowing
*which* terms were glossed in each reference.

**Fix:** `prepare_data.py` rebuilds all three splits by matching each
record back to its source CSV row (joined on `(en, pcm_augmented)`,
disambiguating the 78 rows with duplicate augmented text by consuming
matches in order) and adds `id`, `pcm_clean`, `pcm_augmented`,
`glossed_terms`, `n_glosses`, `subject`. Recovery was exact: 26,232/26,232
rows matched, 0 unclaimed, 0 overlap -- so all runs remain comparable
across the metadata-enrichment boundary.

### 3.4 chrF++ was mislabeled

**Symptom:** not a crash -- a metric-correctness bug. The original code
called `evaluate.load("chrf")` at its default `word_order=0`, which
computes **plain chrF**, and reported it as `"chrf"` in the output dict
while a separate literature convention calls chrF++ specifically
`word_order=2`.

**Fix:** now computes and reports both `chrf` (`word_order=0`) and
`chrf++` (`word_order=2`) explicitly, in both `train.py`'s
`compute_metrics` and `glossary_metrics.py`.

### 3.5 DDP test-set padding broke output writing

**Symptom:** training and test-set evaluation both completed
successfully -- `metrics.json` was written with correct scores -- but the
run then crashed with `ValueError: All arrays must be of the same
length` while writing `test_predictions.csv`, and every downstream
artifact (`glossary_report.json`, `per_term_glosses.csv`) was lost.

**Cause:** under DDP, the distributed sampler pads the test set so it
divides evenly across ranks and per-device batch size (2,623 examples →
2,624 predictions, one padding row appended). `pandas.DataFrame` rejected
constructing a frame from columns of unequal length (2,624 predictions vs
2,623 source/reference/id columns).

**Fix:** predictions are now trimmed to the record count before the
`DataFrame` is built, with a hard failure (not a silent truncation) if
predictions are ever *shorter* than records, since that would silently
mis-pair every subsequent row.

**Recovery:** this bug hit the first afrimt5 run after roughly 2.5 hours
of successful training. Rather than retrain, `evaluate_model.py` (new)
loads the already-saved model and regenerates test predictions and the
glossary report standalone, in ~25 minutes. This script is now the
general-purpose way to (re-)score any completed model without retraining
-- used again for the metric/hyperparameter re-runs in §5.

### 3.6 Missing checkpoint recovery

**Symptom:** not a crash -- a resilience gap noticed after the above
incident. Checkpoints were being written every N steps but
`trainer.train()` was never called with `resume_from_checkpoint`, so an
interrupted run had no way to resume -- every failure cost the entire
elapsed training time.

**Fix:** added `--resume` (bare form finds the latest checkpoint in the
output directory; a path argument resumes from a specific one).

### 3.7 Missing Python.h broke generation (blocking, largest single loss)

**Symptom:** training ran cleanly for exactly one epoch (1,312 steps,
~23 minutes) and then crashed at the first evaluation:
`fatal error: Python.h: No such file or directory`, inside a `gcc`
subprocess invoked from Triton.

**Cause:** PyTorch 2.13's op dispatch routes certain ops -- specifically
hit during `generate()`, not during the training forward/backward pass --
to a Triton-compiled kernel (`bmm_outer_product`), which JIT-compiles a
small C helper (`cuda_utils.c`) at first use. This requires the CPython
development headers (`Python.h`), which `python3.12-dev` provides.
That package was not installed, and this machine has no passwordless
`sudo`, so `apt install` was not an option.

Confirmed by direct trace: `Seq2SeqTrainer.evaluate` →
`model.generate()` → `_prefill` → `sdpa_attention_forward` → torch's
native op registry → `bmm_outer_product/triton_impl.py` → `gcc`
subprocess → failure. Training itself never reaches this code path,
which is why it survived a full epoch before failing -- this bug would
have hit **every model in the sweep** at its first evaluation had it not
been caught here.

Also tested and ruled out: `attn_implementation="eager"` fails
identically, since the Triton dispatch sits below the attention
implementation choice, not inside it.

**Fix:** downloaded the `python3.12-dev` / `libpython3.12-dev` `.deb`
packages directly (`apt-get download`, no root required), extracted them
to `~/.local/include` (both the `python3.12/` headers and the
multiarch-specific `x86_64-linux-gnu/python3.12/pyconfig.h` that
`Python.h` transitively requires), and exported `CPATH` to point `gcc`
at them. This is now done automatically by `run_train.sh` for every run.
Verified directly: 5-beam generation on GPU succeeds with `CPATH` set,
fails identically without it.

**Cost:** this bug cost one full training run (~23 minutes) before being
caught and fixed.

### 3.8 Epoch-boundary saving made crashes maximally expensive

**Symptom:** not itself a bug, but a design choice that amplified the
cost of every other failure -- the original config saved and evaluated
only at epoch boundaries (`eval_strategy="epoch"`), so any crash between
boundaries lost all progress since the last one (up to ~1,312 steps,
~23–150+ minutes depending on model size).

**Fix:** switched to step-based saving (`--save-steps`, default 500),
which bounds the cost of a crash to at most 500 steps (a few minutes) at
any model size. Because `load_best_model_at_end` requires save and eval
cadence to match, this necessarily also switched evaluation to every 500
steps. To keep that affordable, in-training evaluation now runs on a
**seeded 1,000-example shuffled subsample** of dev (full 2,623-example
dev is used nowhere during training) -- shuffled because the splits
inherit the source CSV's subject-grouped ordering, so an unshuffled head
slice would have been almost entirely Computer science. Final test
scoring always uses the complete, un-subsampled test split. This also
required raising `EarlyStoppingCallback`'s `early_stopping_patience` from
its epoch-era value (2, meaning 2 epochs) to an evaluation-count value
recalibrated for the new cadence (4, later 6 -- see §5).

### 3.9 `ddp_find_unused_parameters=False` broke M2M-100

**Symptom:** `RuntimeError: Expected to have finished reduction in the
prior iteration before starting a new one` -- a DDP gradient-reduction
error, immediate (step 2) under M2M-100, never seen under any T5-family
model.

**Cause:** the training config set `ddp_find_unused_parameters=False` as
a throughput optimization, on the assumption ("no parameter is skipped in
a seq2seq forward pass") that held for every T5-family model tested but
does not hold for M2M-100, which has parameters that do not participate
in every forward pass.

**Fix:** default flipped to `True` (safe across architectures, small
extra graph-traversal cost per step), with a per-model override back to
`False` for the three T5-family configs already confirmed to train
cleanly under it (afriteva, mt5, afrimt5).

### 3.10 `forced_bos_token_id` on `model.config` is a hard error in transformers 5.x

**Symptom:** m2m100 crashed at its first evaluation (step 500) with
`ValueError: You have modified the pretrained model configuration to
control generation We detected the following values set -
{'forced_bos_token_id': 49}. This strategy to control generation is not
supported anymore.`

**Cause:** the fix in §3.12 set `forced_bos_token_id` on **both**
`model.config` and `model.generation_config`, following older
`transformers` convention.
transformers 5.x hard-errors if this attribute is set on `model.config`
at all, even when `generation_config` is also set correctly -- the two
were not simply redundant, one is actively forbidden.

**Fix:** set `forced_bos_token_id` only on `model.generation_config`,
never on `model.config`. Verified directly: `model.generate()` succeeds
with only `generation_config` set, and the fix was confirmed to clear
the exact evaluation step that crashed before, on a fresh run.

**Cost:** ~6 minutes of re-training (crash occurred before the first
checkpoint save, since save happens after evaluation completes).

### 3.11 M2M-100's language token was silently wrong (result-invalidating)

**Symptom:** not a crash -- the run completed and produced the *best*
glossary-accuracy score of any model so far (73.18, vs. afrimt5's 56.09).
That result is invalid: manual inspection of `test_predictions.csv`
showed 97.0% of predictions (2,545/2,623) begin with the literal text
`"en "` -- e.g. `"en note say race and color no dey on di list of
acceptable (dem worthy of acceptance or satisfactory) bfoqs"`.

**Cause:** M2M-100's real language-control tokens are internally
formatted as `__en__`, `__af__`, etc., distinct from the plain-string
codes (`en`, `af`, …) used as *keys* into `tokenizer.lang_code_to_id`.
The code in §3.12 resolved the target-language id with
`tokenizer.convert_tokens_to_ids("en")`, which does not raise or return
`unk` -- it silently matches an unrelated ordinary vocabulary entry
(token id 49, some ordinary subword) instead of the real control token
(id 128022). Both NLLB and mBART-50 were checked immediately after this
was found and are **not** affected: NLLB's codes (`tpi_Latn`) are
themselves genuine standalone vocabulary tokens, and mBART-50 exposes
the correct `lang_code_to_id` mapping in agreement with
`convert_tokens_to_ids`. This is specifically an M2M-100 tokenizer
quirk, not a pattern across the three.

The existing validation guard (`token_id == tokenizer.unk_token_id`)
did not catch this, because the wrong id was a legitimate,
non-`unk` vocabulary entry -- just the wrong one.

**Why the numbers still looked good:** the wrong id (49) was used
consistently for every training example and every generated prediction,
so the model learned a fully self-consistent mapping -- it just was not
actually being told "generate English/Pidgin" by anything meaningful.
Translation and glossing quality genuinely came from the parallel data,
with token 49 acting as an arbitrary constant prefix the model learned
to route around, not as language conditioning. The BLEU/chrF/TER scores
this produced (65.85 / 79.15 / 38.59) are not necessarily wrong in an
absolute sense, but every prediction is contaminated with a leaked "en "
prefix -- unpublishable as-is, and the run must be treated as invalid
regardless of the numeric scores.

**Fix:** target-language token resolution now prefers
`tokenizer.lang_code_to_id[TARGET_LANG]` when the tokenizer defines that
mapping (M2M-100 and mBART-50 both do), falling back to
`convert_tokens_to_ids` only when it doesn't (NLLB, in this
`transformers` version). Verified directly against all three tokenizers
post-fix: NLLB → 256179, mBART-50 → 250004, M2M-100 → **128022** (was
49), all confirmed correct against the tokenizer's own `lang_code_to_id`/
`get_lang_id` ground truth.

**Handling:** the flawed run's full artifacts were preserved
(`baseline_runs/output_m2m100/`,
`results_baseline_wrong_lang_token/m2m100/`) for the record, then
m2m100 was requeued with the fix. This is the **third distinct bug**
specific to getting M2M-100 to train correctly (see also §3.9, §3.10),
which itself may be worth a footnote -- this model needed
disproportionately more debugging than any other in the sweep.

### 3.12 Language-token gap in all three multilingual MT models

Not a bug -- a modelling decision forced by a fact about the pretrained
vocabularies, checked directly rather than assumed:

| Model | Language codes in vocabulary | `pcm` present? |
|---|---|---|
| NLLB-200-distilled-600M | 202 (includes Hausa, Igbo, Yoruba) | **No** |
| mBART-50 | 52 | **No** |
| M2M-100 | 100 | **No** |

None of the three multilingual MT models support Nigerian Pidgin as a
named target language, so fine-tuning must repurpose an existing language
token:

- **NLLB → `tpi_Latn`** (Tok Pisin). Chosen because it is the only
  English-lexified creole in NLLB's 202-language inventory, and unlike
  reusing `eng_Latn` it gives the decoder a target token distinct from
  the source language, which avoids biasing generation toward copying
  the English input verbatim.
- **mBART-50 → `en_XX`**, **M2M-100 → `en`** (English). Neither
  inventory contains any creole, so no better proxy exists.

A related silent-failure risk was also fixed: the original code never
set `forced_bos_token_id`, meaning generation for these three models
would not be pinned to the intended target-language token at all and
could decode into whichever language the model defaults to. This is now
set from the resolved target-language token id at model-load time.

**This is a stated methodological limitation, not a detail** -- any paper
reporting NLLB/mBART/M2M-100 numbers on this benchmark should disclose
the proxy-token substitution explicitly, since it is a confound relative
to models that see no equivalent language-identity conflation.

### 3.13 The sweep-chaining watcher does not survive a session restart (19h46m idle GPUs)

**Symptom:** `nllb` finished cleanly at 2026-08-15 20:28 UTC. The next
model in the chain (a corrected m2m100 re-run) did not start. Both GPUs
sat idle until this was noticed at 2026-08-16 16:14 UTC -- **19 hours
46 minutes** of idle compute.

**Cause:** the automated model-to-model chaining (§3.11's "requeue")
was implemented as a shell script that polls a specific process ID and
launches the next model when it exits, run via the coding assistant's
own background-task tracking. That tracking is scoped to the assistant
session; when the session was torn down and restarted (for reasons
outside this project -- an unrelated tool/agent lifecycle event), the
watcher process was killed with it. Training jobs themselves were
unaffected -- `run_train.sh` launches them with `setsid` under `nohup`
and they have no controlling terminal or session dependency, which is
exactly why `nllb` ran to completion regardless. The watcher script
had the *same* `nohup`+background-launch pattern applied to the models
it starts, but the watcher's own process was started under the
assistant's tracked background-task mechanism, not detached the same
way, and did not survive.

**Fix:** the corrected m2m100 → nllb (rerun) → mbart50 chain was
re-launched with the watcher script itself started via `nohup ... &
disown` directly at the shell level, matching the pattern already
proven to survive a session boundary (`run_train.sh`'s own launches).
This is expected to be more robust, but has not yet been proven across
a second restart -- if GPUs are found idle with a chain incomplete
again, check `pgrep -af chain_` before assuming a run is still in
progress.

**Cost:** none to data or model quality -- the completed run (`nllb`)
was unaffected and its results stand. Cost was purely idle GPU-hours.

### 3.14 Chain watcher raced past a still-running job on a stale PID

**Symptom:** the chain queued to run `mt5_large` → `cheetah` → `toucan`
logged all three as "finished" within about 40 seconds of launching
`cheetah`, and `toucan` "finished" 20 seconds after that -- while
`mt5_large`'s actual training process was still running, 4 minutes into
what would be a multi-hour job (confirmed alive throughout: still
training, no errors, unaffected, at the moment this was discovered).

**Cause:** the wait-loop was given a PID for `mt5_large` captured via a
separate `pgrep` call made ~90 seconds after that model was launched in
an earlier, different shell invocation. That PID was stale -- not the
actual long-lived `torchrun` process still running minutes later
(`torchrun`'s own fork/exec chain means a PID sampled shortly after
launch is not guaranteed to be the one still alive later). The watcher
believed `mt5_large` had already exited and immediately launched
`cheetah`, which itself exited almost instantly (§3.15), and then
`toucan`, which also exited almost instantly (§3.16) -- both for reasons
unrelated to this bug, which is what makes the race hard to notice from
the log alone: three "finished" lines in under a minute look wrong on
their face, but the *individual* failures were real and would have
happened at any time.

**Consequence:** `cheetah` and `toucan` briefly attempted to start
their own `torchrun`/NCCL process groups on the same two GPUs
`mt5_large` was actively using. Both crashed at data/tokenizer loading,
before requesting any real GPU memory or joining a process group, so no
corruption or resource contention reached `mt5_large` -- confirmed by
its continued clean, error-free training afterward. This was luck of
timing (both failed fast, at import/download time), not a property of
the fix.

**Fix:** never capture a PID for a job in a separate command from the
one that launched it. The `run_and_wait` pattern used successfully in
§3.13 (capturing `$!` in the exact same shell as the `nohup ... &`) is
correct and was not the source of this bug -- the hardcoded PID for the
*first* model in a chain that had already been launched earlier was.
The corrected chain instead re-polls by command pattern
(`pgrep -f "torchrun --standalone.*model $model"`) every iteration
rather than trusting any single PID snapshot, for every model in the
chain, including one already running before the chain script starts.

### 3.15 UBC-NLP/cheetah-1.2B is a manually-gated HF repo

**Symptom:** immediate `401 Unauthorized` / `GatedRepoError` on the
very first download attempt (`config.json`), before any training code
ran.

**Cause:** confirmed via the HF API (`gated: manual` on
`UBC-NLP/cheetah-1.2B`) -- this repo requires the requesting account to
apply for access and be manually approved by the repo owner, then
authenticate downloads with an `HF_TOKEN` that has that access.

**Status: resolved.** Access was requested and approved on the
HuggingFace account this benchmark runs under, and an access token was
supplied and stored in the standard `huggingface_hub` credential cache
(`~/.cache/huggingface/token`, file permissions restricted to the
owning user) so every download in this pipeline authenticates
automatically without the token appearing in any script, log, or
report. Verified directly: `hf_hub_download` on a previously-401'd file
now succeeds. `cheetah` also turned out to need the same tokenizer fix
as `toucan` (§3.16) -- same lab, same underlying bug -- which was caught
and verified *before* re-queueing it, not discovered mid-run. It is
back in the active chain, after `mt5_large` and `toucan`.

### 3.16 UBC-NLP toucan-1.2B and cheetah-1.2B: declared tokenizer class does not match the tokenizer file

**Symptom:** `TypeError: argument 'vocab': 'dict' object cannot be
converted to 'Sequence'`, inside `transformers`' T5 tokenizer
construction, immediately on `AutoTokenizer.from_pretrained` -- before
any training code ran. Reproduced in isolation (not DDP-specific),
confirmed with both `use_fast=True` and `use_fast=False`, and confirmed
**identical on both `toucan-1.2B` and `cheetah-1.2B`** (§3.15) once
`cheetah` became accessible -- same tokenizer_class declaration, same
actual BPE format, same fix. Both are UBC-NLP releases, evidently from
the same export pipeline.

**Cause:** each repo's `tokenizer_config.json` declares
`"tokenizer_class": "T5Tokenizer"`, which routes loading through T5's
SentencePiece/Unigram-specific conversion logic. But each repo's actual
`tokenizer.json` has `model.type == "BPE"` -- a plain BPE vocabulary,
not the Unigram format T5's conversion path expects to build a
`Sequence` from. This is a mismatch in the repos' own published files,
not a local environment or library-version problem: `tokenizer.json`
loads correctly and produces sensible tokenizations completely on its
own once the T5-specific class resolution is bypassed.

**Fix:** for both models, the tokenizer is loaded directly as
`PreTrainedTokenizerFast(tokenizer_file=...)` from a `hf_hub_download`
of `tokenizer.json`, with `pad_token="<pad>"`, `eos_token="</s>"`,
`unk_token="<unk>"` set explicitly (read from each repo's
`special_tokens_map.json` -- identical values on both) -- bypassing
`AutoTokenizer`'s class resolution entirely rather than trying to fix
the T5 conversion path.

**Verified end-to-end before relaunching training** on both, not
assumed to transfer just because the bug looked identical:

- `toucan`: tokenizer loads (vocab size 250,102), encodes correctly,
  and a full forward pass through the pretrained (not yet fine-tuned)
  model generates coherent zero-shot Pidgin -- *"the cell membrane dey
  control wetin dey inside the cell"* for "the cell membrane controls
  what enters the cell" -- a good sign for the fine-tune to come.
- `cheetah`: tokenizer loads identically (same vocab size, same
  pad/eos ids), but zero-shot generation produces only
  `<extra_id_0></s>` -- an mT5 span-corruption sentinel followed by
  immediate end-of-sequence, not fluent text. This is not evidence of
  a broken fix: it is the expected behavior of a base checkpoint that
  was pretrained on span-corruption but not translation-tuned before
  release, unlike `toucan` which already responds to a translation-style
  prompt zero-shot. The tokenizer and model both load and run without
  error, produce valid in-vocabulary token ids, and require no further
  changes -- whether fine-tuning teaches it to translate is what the
  actual training run answers, the same as for every other model in
  this sweep.

### 3.17 toucan's first fine-tuning run failed silently: fragmented prefix token, and/or too-aggressive LR

**Symptom:** no crash, no NaN, no error at any point. The run completed
65% of its schedule (4,275/6,560 steps, ~5h37m) with `eval_loss`
essentially flat around 3.0–3.9 the whole way (every other model in
this sweep reached <0.9 within the first 500 steps) and `eval_bleu`
stuck under 1.0 throughout (step 500: 0.06; step 3500: 0.80 -- for
comparison, every other model exceeded BLEU 30 by its first
evaluation). `eval_gen_len` oscillated wildly between evaluations (14,
90, 66, 49, 57, 65, 42, 39) rather than stabilizing.

Reading actual generated predictions (not just the aggregate scores)
showed why the metrics were this bad: the model produced fluent,
grammatical Pidgin that was **unrelated to the source sentence** --
e.g. source *"Na one student leki (1995)discuss adaptive transfer"* →
prediction *"Di researcher don get two different ways to do research"*.
This is a decoder generating a plausible-sounding continuation while
not conditioning on the encoder output, not a model that "hasn't
learned yet."

**Two candidate causes, both plausible, addressed together rather than
isolated one at a time given the cost of another multi-hour run:**

1. **The configured source prefix, `"<2pcm> "`, was never a real token
   in this checkpoint's vocabulary.** Checked directly against
   `tokenizer.json`'s `added_tokens` list: 103 entries total
   (`<pad>`, `</s>`, `<unk>`, and 100 T5 sentinel tokens
   `<extra_id_0>`–`<extra_id_99>`) -- no language-tag tokens of any
   kind. `<2pcm>` silently fragments into five ordinary BPE pieces
   (`▁<`, `2`, `p`, `cm`, `>`) prepended as noise to every input. This
   prefix choice was made early in this project by analogy to a
   generic `<2xx>` multilingual-NMT convention and was never checked
   against this specific checkpoint's actual vocabulary (§3.16 checked
   the tokenizer *loads*, not that this particular prefix is
   meaningful to it -- a narrower verification than it should have
   been). Note this is not a full explanation on its own: the
   pre-launch zero-shot sanity check in §3.16 used this same
   fragmented prefix and produced accurate, on-topic output, so
   fragmentation alone does not obviously explain total training
   failure.
2. **Learning rate 1e-3 (Adafactor), identical to `afriteva_v2_large`
   and `mt5_large`, both of which converged cleanly** -- same
   optimizer, same pipeline code, same effective batch, run back to
   back on the same hardware. `toucan` is the outlier at this setting,
   which points at something specific to this checkpoint's pretrained
   weight scale rather than a problem with the 1e-3/Adafactor
   combination in general.

**Fix:** both changed together. Source prefix switched to
`"translate English to Pidgin: "` -- the exact string already proven
across five other T5-family models in this sweep, rather than any
`<2xx>`-style tag. Learning rate pulled back to 1e-4. Neither change is
independently confirmed as *the* fix (both were changed at once, given
the cost of isolating them one at a time on a 1.2B model); the
combination's effect is what the re-run's first evaluations test.

**Handling:** the failed run was stopped (not left to finish
uselessly) via the same graceful `run_train.sh` shutdown path used
elsewhere in this pipeline (§3.8) -- confirmed both GPUs freed, no
orphaned processes. Its predictions and checkpoints are preserved at
`baseline_runs/output_toucan/` and
`results_baseline_wrong_prefix_lr1e3/toucan/` (metrics.json and
glossary_report.json do not exist for this run, since those are only
written at successful completion -- the preserved evidence is the
mid-training `predictions.csv` and the full training log). Re-launched
with the fix and **confirmed at the first evaluation** (step 500):
BLEU 38.82, chrF++ 55.93, eval_loss 1.23 -- against the failed run's
BLEU 0.06, chrF++ 8.28, eval_loss 3.93 at the identical step. This is
now in line with `mt5_large` and `afriteva_v2_large`'s own first
evaluations rather than an outlier. Full results in §7.7 -- the run
ultimately completed successfully, though not without a second
unexplained incident along the way (§3.18).

### 3.18 The corrected toucan run was killed by an external SIGTERM at 66% and restarted from scratch, cause unidentified

**Symptom:** the corrected run (§3.17), confirmed healthy at its first
evaluation, continued training normally through step 4,361/6,560 (66%,
~5h41m elapsed) and then received `SignalException: Process ... got
signal: 15` (SIGTERM) with no preceding error, warning, or resource
exhaustion in its own log. A new, completely independent `torchrun`
process for `toucan` started roughly one minute later, from a fresh
`STARTING TRAINING: toucan` at step 0 -- not `--resume`, so the first
run's 66% of progress (though its step-4000 checkpoint remained on
disk) was discarded rather than continued from.

**Cause: not identified.** Ruled out, specifically: this was not
caused by anything in this pipeline's own tooling. The only chain
script alive at the time (`chain_toucan_to_cheetah.sh`) only ever
*waits* for `toucan` and launches `cheetah` on completion -- it
contains no code path that could launch `toucan` a second time, and
its own log shows it sitting in the wait loop throughout, only
proceeding to `cheetah` after the *second* run's completion. No other
chain or watcher script from earlier in this session was still running
(checked directly). The signal was sent by something outside this
pipeline's own processes -- most plausibly another process or user on
this shared machine (multiple concurrent logins were observed on this
host earlier in this project, §1's environment was never confirmed to
be single-tenant), though this is inference, not confirmed root cause.

**Consequence:** purely a compute-time cost, not a correctness one --
roughly 5h41m of redundant training. The eventual full second run
(§7.7) completed cleanly with results in line with expectations, so
nothing about this incident invalidates `toucan`'s final numbers.

**Handling:** none needed beyond noting it. The re-poll-by-pattern
chain design (§3.14) meant this externally-caused interruption did not
require any manual intervention to route around -- the chain simply
kept waiting until a `toucan` process matching the expected pattern no
longer existed and `output_toucan/glossary_report.json` was present,
which is exactly what happened once the second run finished. Flagged
here as a limitation on how autonomously this pipeline can be trusted
to run unattended: it is robust to *its own* process-management bugs
(§3.13, §3.14) but not to external interference, which it cannot
detect or explain, only survive.

### 3.19 cheetah reproduced toucan's exact failure at 1e-3 -- with a correct prefix, isolating the true cause

**Symptom:** once the chain reached `cheetah` (its config still at the
original 1e-3 -- only `toucan`'s config had been fixed, §3.17), it
showed the identical failure signature: `eval_bleu` near zero across
four evaluations (0.02, 0.0003, 0, 0.18), `eval_loss` stuck between
3.8 and 6.3 with no clear downward trend, and wildly erratic
`eval_gen_len` (199, 5, 3, 118 across four consecutive evaluations).

**This is the evidence that isolates the root cause.** `cheetah`'s
source prefix was `"translate English to Pidgin: "` from the very
start -- the same proven string used successfully by five other
models -- never the broken `<2pcm>` tag that `toucan` had. Two
different checkpoints, two different prefixes (one broken, one
already correct), the same 1e-3 Adafactor rate, the same failure. The
prefix fragmentation theorized in §3.17 as a *possible* contributing
factor for `toucan` is therefore not the operative cause here, and by
extension was likely never more than a secondary issue there either:
**1e-3 Adafactor is the common thread, and the actual cause, for both
UBC-NLP checkpoints specifically** (afriteva_v2_large and mt5_large,
from different labs, tolerated 1e-3 without incident).

**Fix:** learning rate pulled back to 1e-4, identical to `toucan`'s
fix, with `cheetah`'s already-correct prefix left untouched. Preserved
the failed run first (`baseline_runs/output_cheetah/`,
`results_baseline_lr1e3/cheetah/`). Re-launched and **confirmed
working** at the first evaluation: BLEU 36.66/chrF++ 55.81 at step
500, against the failed run's BLEU 0.02/chrF++ 6.30 at the identical
step -- in line with toucan's own fixed-run first evaluation (§3.17).
Convergence has continued cleanly since, no instability at any point:

| Step | Epoch | Loss | BLEU | chrF++ |
|---|---|---|---|---|
| 500 | 0.38 | 1.295 | 36.66 | 55.81 |
| 1000 | 0.76 | 0.936 | 46.71 | 61.06 |
| 1500 | 1.14 | 0.774 | 53.64 | 67.32 |
| 2000 | 1.52 | 0.667 | 58.69 | 69.84 |

Completed cleanly with no further incidents; full results in §7.8 --
final GlossF1 73.16, AfriCOMET 71.84 (§10.3).

### 3.20 `tie_word_embeddings` silently flipped on save for 6 of 12 models -- checkpoints reloaded as fluent-looking garbage, flagship model included

**Found while building a validation-set leaderboard** (§9.5's model-
selection correction) rather than during the original sweep, since the
original test-set numbers were computed from the in-memory model
straight out of `Trainer.train()` and never required a checkpoint
reload -- this bug only manifests on reload, so it was invisible until
one was actually attempted.

**Symptom**: re-scoring `mt5_large`'s saved checkpoint on the dev split
produced pure repetition garbage (`'钽钽钽钽钽钽...ittimetittimet...'`),
not a crash, not a plausible-but-worse translation -- at BLEU/GlossAcc
0 for every one of `mt5`, `afrimt5`, `mt5_large`, `afriteva_v2_large`,
`cheetah`, `toucan`. The other 6 models (`afriteva`, `m2m100`,
`m2m100_1.2b`, `nllb`, `mbart50`, `seamless`) reloaded and scored
correctly.

**Diagnosis, ruled out in order rather than guessed**: not the tokenizer
(encoding a test string through the saved vs. the fresh upstream
tokenizer produced byte-identical token IDs); not corrupted weights (no
NaN/Inf in any of the 557 saved parameter tensors, checked directly);
not a batching/padding-side artifact (reproduced with a single example,
batch size 1); not beam-search instability (reproduced identically at
greedy decoding); not a general environment/`transformers`-version
issue (a *fresh* `google/mt5-large` load, same code path, produced
coherent if task-naive English text, not garbage).

**Root cause**: every one of the 6 broken models has a real, upstream-
documented **untied** input/output embedding architecture
(`tie_word_embeddings: False` in the original HF config -- a T5 v1.1/
mT5 characteristic; confirmed directly against all 12 checkpoints'
real upstream configs, not assumed by architecture family). Somewhere
inside the `Seq2SeqTrainer`/`accelerate` save path, this flag is
silently overwritten to `True` in the saved `config.json` for exactly
these 6 (the other 6 already default to `True`, which happens to be
correct for them, masking the same underlying drift). The actual
fine-tuned `lm_head.weight` tensor is still present and untouched in
`model.safetensors` (confirmed via `safetensors.safe_open` -- both
`lm_head.weight` and `shared.weight` exist as separate keys). But
`tie_word_embeddings: True` tells `from_pretrained()` to *tie* the two
at load time, silently discarding the correct saved `lm_head.weight`
in favor of a copy of the input embeddings, which were never optimized
to function as an output projection. Generation still runs -- no
missing/unexpected-key warning, no error -- it just runs on a
functionally-random output layer, producing exactly this kind of
fluent-shaped nonsense rather than a crash.

**Blast radius**: this affects any future reload of these 6 saved
checkpoints, which is to say every use beyond the original training
run -- **including both `toucan` (the finalized PidginEdu-LLM
flagship, §9.5) and `mt5_large` (the automated-metric leader)**.
Uploading any of these 6 to HuggingFace, running them in the planned
inference demo, or redistributing them as-is would all have silently
produced garbage output. The original test-set numbers reported
throughout this document for these 6 models are unaffected
(computed from the in-memory model during training, never a reload),
but the saved artifacts backing those numbers were broken until fixed
here.

**Fix, no retraining required**: since the real weights were never
lost, this is a one-field correction. Patched `tie_word_embeddings`
back to `False` in `config.json` for all 6 affected models -- both the
root checkpoint directory and their intermediate `checkpoint-6500`/
`checkpoint-6560` subdirectories (12 files total) -- and verified each
of the 6 regenerates fluent, correctly-glossed Pidgin output
afterward (e.g. `mt5_large`: `"Di constitution dey draft (na written
order to bank or persin to pay specified money), di delegates know say
dem wanted (wetin person desire / person wey police dey find) george
washington to be president."`). Confirmed LoRA checkpoints are
unaffected -- adapters reference `base_model_name_or_path` and always
reload the base model fresh from the HF hub, never a locally-saved
config, so they never see the corrupted value. Added a safeguard in
`train.py`: the model's true `tie_word_embeddings` value is captured
immediately after loading (`ORIGINAL_TIE_WORD_EMBEDDINGS`), and
directly re-asserted in the saved `config.json` right after
`trainer.save_model()` for every future full-fine-tune run, with a
warning logged if a drift is ever detected -- correct regardless of
which internal Trainer/accelerate step actually causes it, since the
root cause inside that stack was not isolated further.

---

## 4. Glossary-accuracy metric

Implemented in `glossary_metrics.py`. This is the benchmark's novel
contribution, so its design choices are documented in enough detail to
be defended in review.

### 4.1 What it measures

Standard MT metrics (BLEU, chrF) cannot distinguish "reproduced the right
terminology gloss" from "produced fluent text around it." For every term
a reference glosses, the metric checks two independent things -- whether
the system glossed the term at all, and whether the gloss it produced
means the right thing -- and reports four numbers:

| Metric | Formula | Interpretation |
|---|---|---|
| Gloss Presence Rate (GPR) | glossed / expected | Did the model attempt to gloss the term (recall of the *behavior*) |
| **Gloss Accuracy (GA)** | correct / expected | **Headline metric** -- glossed *and* content matches |
| Gloss Precision (GP) | correct / produced | Of every gloss emitted, how many were right (penalizes over-glossing) |
| Gloss F1 | harmonic mean(GA, GP) | Balances the above two |

### 4.2 Design decisions

**Term inventory, not "every parenthetical."** A parenthetical is only
treated as a gloss if the word immediately preceding it is in the
corpus-wide glossary term inventory (2,473 terms, built from all three
splits -- a fixed public artifact released with the benchmark, not
leakage from test references). Without this filter, non-gloss
parentheticals in the source text (`new york (cuny)`,
`(chronic adrenal insufficiency)`) would be miscounted as glosses.

**Content matched by chrF similarity ≥ 50, not exact string.** Pidgin
orthography varies within the corpus itself (`dem`/`den`, `de`/`di`), so
exact-match scoring would undercount correct glosses that are
semantically identical but spelled differently. Threshold is
configurable (`--gloss-threshold`).

**132 of 2,858 test terms (4.6%) are excluded as unscorable.** These are
listed in a row's `glossed_terms` but the reference's own inline gloss
for them is not parseable by the same extraction pattern (inflected
form, unusual spacing). Excluding them is what allows a perfect system
(hypothesis = reference verbatim) to score exactly **100.0**, not a
lower ceiling -- verified directly (§4.4). The exclusion count is reported
in every output, so it is auditable rather than hidden.

**Four numbers, not one, specifically to resist gaming.** GA alone can be
inflated by a system that glosses indiscriminately. This was verified
with a synthetic adversarial test case: a system that appends a spurious
gloss to every sentence scores GA 100.0 (never misses an expected term)
but GP 50.8 (half its glosses are on terms that shouldn't be glossed),
giving F1 67.3 -- correctly penalized despite perfect recall.

### 4.3 Dual-reference MT scoring

Because training targets are glossary-augmented, raw BLEU/chrF against
those targets is inflated relative to (and not comparable with) standard
English-Pidgin MT literature. Every model is therefore scored twice:

- **`augmented_reference`** -- hypothesis vs. `pcm_augmented` as generated.
  Only meaningful within this benchmark.
- **`clean_reference`** -- inline glosses stripped from both hypothesis
  and reference (`strip_glosses()`, using the same term-inventory logic
  as extraction) before scoring. This is the number that is comparable
  to other English-Pidgin MT work and is what `aggregate_results.py`
  reports as the headline BLEU/chrF++/TER.

### 4.4 Validation of the metric itself

Before trusting it on real model output, the metric was run against five
synthetic systems built from known ground truth:

| Synthetic system | GA | GPR | GP | F1 | Purpose |
|---|---|---|---|---|---|
| Perfect (reference verbatim) | **100.0** | 100.0 | 99.16 | 99.58 | Ceiling check |
| Never glosses (clean text only) | 0.04 | 0.04 | 5.0 | 0.07 | Floor check |
| Glosses every term, wrong content | 0.0 | 100.0 | 0.0 | 0.0 | Content-check isolation |
| Half glossed, half not | 51.1 | 51.1 | 98.65 | 67.3 | Partial-credit sanity |
| Over-glosses (spurious extra glosses added) | 100.0 | 100.0 | 50.76 | 67.34 | Precision penalizes gaming |

The GP=99.16 (not 100) on the "perfect" row, rather than an exact match,
reflects real chrF variance from the similarity threshold, not a bug.

---

## 5. Learning-rate / early-stopping incident: why glossing initially failed

This is the most consequential empirical finding to date and is
documented in full because it materially changes how the remaining
T5-family models in the sweep should be configured.

### 5.1 Original results (all three models, original hyperparameters)

| Model | LR | Patience | BLEU (clean) | chrF++ (clean) | test loss | **GlossAcc** | Glosses produced / expected |
|---|---|---|---|---|---|---|---|
| afriteva | 3e-5 | 4 | 57.71 | 74.38 | 0.604 | **41.71** | 1,914 / 2,726 |
| mt5 | 5e-5 | 4 | 58.49 | 75.53 | 1.164 | **0.00** | 29 / 2,726 |
| afrimt5 | 3e-5 | 4 | 61.28 | 77.10 | 0.922 | **0.18** | 42 / 2,726 |

afrimt5 had the **best BLEU of the three** while producing essentially
zero correct glosses -- a model that had clearly learned to translate but
not to gloss.

### 5.2 Diagnosis

Ruled out, in order, before accepting the real explanation:

1. **Metric bug?** No -- afriteva scored 41.71 on the identical metric
   code and data, so the metric correctly detects glossing when it is
   present.
2. **Label truncation at `max_target_length=200`?** No -- directly
   measured: only 0.0–0.3% of training targets exceed 200 tokens for any
   of the three tokenizers. (mT5 in fact tokenizes Pidgin *more*
   efficiently than afriteva's tokenizer: mean 44 vs 58 tokens per
   target.)
3. **Real underfitting**, confirmed by two independent signals:
   - Final test loss was 0.92 (afrimt5) and 1.16 (mt5) vs. afriteva's
     0.60 -- both mT5 models simply hadn't converged as far.
   - `mt5`'s baseline run **early-stopped at step 3500 of 6560** (best
     checkpoint at step 1500) -- it saw barely half the intended
     schedule before patience-4 (2,000 steps of tolerance at the
     500-step eval cadence) cut it off.
   - Generation length at test time was ~80–83 tokens for both mT5
     models vs. afriteva's 126 -- consistent with "translates, does not
     also generate the gloss span."

Note the LR values were *not* uniform across the failing pair (5e-5 for
mt5, 3e-5 for afrimt5) and afriteva shared afrimt5's exact 3e-5/patience-4
config yet succeeded -- so learning rate alone is not a complete
explanation. It is, however, consistent with a known fact about the T5
pretraining/fine-tuning literature (T5-family models conventionally use
substantially higher fine-tuning rates than BART-style encoder-decoders,
often via Adafactor), and combined with the low early-stopping patience
that was cutting training off before the harder secondary behavior
(glossing) had time to emerge, both are plausible contributing causes.

### 5.3 Change made

Two changes, applied together to `mt5` and `afrimt5` only:

- Learning rate: mt5 5e-5 → **1e-4**; afrimt5 3e-5 → **1e-4**
- Early-stopping patience: 4 → **6** evaluations (3,000 steps of
  tolerance at 500-step cadence, vs. 2,000 before)

The original trained models and their full metrics/predictions were
preserved before re-running (`baseline_runs/`, `results_baseline_lr3e5/`)
specifically so this comparison would remain reproducible.

### 5.4 Result after the change

| Model | GlossAcc before | **GlossAcc after** | BLEU before → after | chrF++ before → after |
|---|---|---|---|---|
| mt5 | 0.00 | **50.11** | 58.49 → 62.65 | 75.53 → 77.79 |
| afrimt5 | 0.18 | **56.09** | 61.28 → 63.41 | 77.10 → 78.19 |

Both models improved on **every** axis simultaneously -- translation
quality and terminology accuracy both rose, with no tradeoff. Both
completed the full 6,560-step schedule (no premature early stopping);
final checkpoints retained were step 6500 and 6560 for both, i.e.
training ran to its natural end rather than being cut off. Training
loss decreased smoothly throughout for both (mt5: 1.33 → 0.55; afrimt5:
1.23 → 0.51), with no instability introduced by the higher learning
rate.

**Implication for the rest of the sweep:** the four remaining
T5-family models (`afriteva_v2_large`, `mt5_large`, `cheetah`, `toucan`
-- the flagship) are currently configured at their original conservative
learning rates and inherit the same risk. This should be corrected
before those runs, not discovered again at 1.2B-parameter cost.

---

## 6. Full hyperparameters, current sweep configuration

All models: fp32, AdamW (`adamw_torch`) unless noted, effective batch
16 (per-GPU batch × gradient accumulation × 2 GPUs), gradient
checkpointing on, 5 epochs (6,560 steps for the current train split
size), save/eval every 500 steps, early-stopping patience 6
evaluations (§5.3), max source/target length 200 tokens, weight decay
0.01, seed 42, `load_best_model_at_end=True` with
**`metric_for_best_model="chrf"`** (plain chrF, `word_order=0` -- not
chrF++). This is worth stating precisely rather than assuming: every
model's checkpoint was selected by plain chrF during training, while
the leaderboard in §7 ranks by chrF++/BLEU/GlossAcc computed after the
fact on that already-selected checkpoint. The two metrics are highly
correlated in practice (§3.4), so this is unlikely to have changed
which checkpoint "won" for any model, but it was never explicitly
verified per-model, and a paper reporting these numbers should name the
actual selection criterion rather than the display criterion. Beam
search: 1 beam (greedy) for in-training validation, 5 beams for the
final test-set generation that produces every number in §7 (§3.8).

| Model | Checkpoint | Params | LR | Per-GPU batch | Optimizer | `ddp_find_unused_parameters` | Status |
|---|---|---|---|---|---|---|---|
| afriteva | castorini/afriteva_base | 229M | 3e-5 | 8 | adamw_torch | False (confirmed safe) | done |
| m2m100 | facebook/m2m100_418M | 418M | 3e-5 | 8 | adamw_torch | **True (required -- see §3.9)** | done (re-run, first run invalidated -- §3.11/§7.1) |
| mt5 | google/mt5-base | 580M | **1e-4** (raised, §5) | 4 | adamw_torch | False (confirmed safe) | done |
| afrimt5 | masakhane/afri-mt5-base | 580M | **1e-4** (raised, §5) | 4 | adamw_torch | False (confirmed safe) | done |
| nllb | facebook/nllb-200-distilled-600M | 600M | **1e-4** (raised, §7.2 -- confirmed: GlossAcc 0.26 → 79.86) | 4 | adamw_torch | True (default, untested) | done (re-run) |
| mbart50 | facebook/mbart-large-50-many-to-many-mmt | 680M | **1e-4** (raised preemptively on nllb's evidence, §7.2 -- confirmed: now overall leader, §7.3) | 4 | adamw_torch | True (default, untested) | done |
| afriteva_v2_large | castorini/afriteva_v2_large | 1B | **1e-3** (raised pre-emptively, §5.4/§10 -- confirmed: essentially tied for 2nd overall, §7.5) | 2 | **adafactor** | True (default, untested) | done |
| mt5_large | google/mt5-large | 1.2B | **1e-3** (raised pre-emptively, §5.4/§10 -- confirmed: best GlossF1 in sweep, §7.6) | 2 | **adafactor** | True (default, untested) | done |
| toucan | UBC-NLP/toucan-1.2B | 1.2B | ~~1e-3~~ → **1e-4** (pulled back after failure, §3.17) | 2 | **adafactor** | True (default, untested) | done (2nd run -- 1st failed at 1e-3, §3.17; interrupted by external SIGTERM once more, §3.18; succeeded on 3rd attempt) |
| cheetah | UBC-NLP/cheetah-1.2B | 1.2B | ~~1e-3~~ → **1e-4** (pulled back after failure, §3.19 -- confirmed: BLEU 0.02 → 36.66 at step 500) | 2 | **adafactor** | True (default, untested) | done (re-run -- first attempt failed identically to toucan's, confirming LR not prefix as the cause, §3.19/§7.8) |
| seamless | facebook/seamless-m4t-v2-large | 1.37B | 1e-4 (pre-emptive default carried over from the nllb/mbart50 fix, §7.2 -- held on the first attempt, §9.6) | 2 | **adafactor** | True (default, untested) | done (§9.6) |
| m2m100_1.2b | facebook/m2m100_1.2B | 1.24B | 1e-4 | 2 | **adafactor** | **True (required -- same M2M-100 family finding as the 418M model, §3.9)** | done (§9.6) |

**`madlad3b` (3B) and `t5-v1_1-xl` (2.85B) have no completed full-FT row above** -- both were attempted, both hit a genuine memory ceiling (§9.3) before completing even a full forward+backward pass at any tested batch size or sequence length, so there is no stable full-FT hyperparameter configuration to report for them. Their only completed result is under LoRA (§11.1's table).
**1B+ tier uses Adafactor, not AdamW.** fp32 AdamW costs ~16 bytes/param
(weights + gradients + two moments); on a 32GB V100 under DDP (each GPU
holds a full, unsharded optimizer copy -- no ZeRO/FSDP sharding is
configured), that puts 1.2B-parameter models at ~19.2GB just for
optimizer state, before activations and the 250k-vocabulary logits
softmax. Adafactor's factored second moments bring this to roughly half,
which is what makes this tier trainable on this hardware at all.

**Originally excluded from this benchmark** (superseded by the
paradigm-diversity roster revision, §9, rather than pursued directly):
- Pidgin-UNMT (220M) -- investigated properly rather than assumed
  unworkable; see §9.1 for the full finding (wrong repo in the
  reference notebook, and the actual pretrained checkpoint no longer
  retrievable from its linked source).
- mT5-XL (3.7B), Cheetah-3.7B (3.7B) -- fp32 AdamW optimizer state alone
  (~59GB) exceeds this hardware's 32GB/GPU even before sharding
  overhead; would require ZeRO-3/FSDP with CPU offload.
- TranslateGemma-4B -- decoder-only architecture (not
  `AutoModelForSeq2SeqLM`-compatible, needs a different training loop),
  gated on HuggingFace (returns 401 without an accepted license + token),
  and bf16-native while this hardware has no bf16 support.

Replaced by MADLAD-400 3B/7B, T5 v1.1-XL, and SeamlessM4T v2 Large --
see §9 for the full rationale and compatibility findings.

---

## 7. Results so far

MT scores below are **clean-reference** (glosses stripped from both
sides) -- the numbers comparable to standard English-Pidgin MT work.
Glossary scores are always computed against the augmented reference,
since that is what they measure.

| Model | BLEU | chrF | chrF++ | TER | GlossAcc | GlossPresence | GlossPrecision | GlossF1 |
|---|---|---|---|---|---|---|---|---|
| **mt5_large** | 66.92 | **82.16** | **80.59** | 32.23 | 79.68 | 80.23 | 77.38 | **78.51** |
| m2m100_1.2b | **68.50** | 81.73 | 80.14 | **30.81** | 77.15 | 77.59 | 77.26 | 77.20 |
| mbart50 | 68.18 | 81.55 | 79.96 | 30.82 | 75.90 | 76.27 | **77.61** | 76.74 |
| afriteva_v2_large | 66.32 | 81.19 | 79.57 | 32.85 | 78.58 | 78.87 | 74.61 | 76.54 |
| nllb | 64.61 | 81.22 | 79.60 | 35.83 | **79.86** | **80.59** | 72.28 | 75.88 |
| seamless | 67.56 | 81.35 | 79.74 | 32.51 | 75.75 | 75.86 | 75.28 | 75.52 |
| toucan | 64.74 | 81.04 | 79.34 | 35.70 | 75.35 | 76.93 | 72.63 | 73.96 |
| m2m100 | 66.38 | 80.35 | 78.66 | 33.45 | 73.15 | 74.80 | 73.42 | 73.28 |
| cheetah | 66.16 | 81.28 | 79.62 | 33.80 | 74.14 | 76.12 | 72.20 | 73.16 |
| afrimt5 | 63.41 | 79.90 | 78.19 | 37.88 | 56.09 | 62.55 | 65.23 | 60.32 |
| mt5 | 62.65 | 79.55 | 77.79 | 38.87 | 50.11 | 57.56 | 63.74 | 56.11 |
| afriteva | 57.71 | 76.97 | 74.38 | 43.54 | 41.71 | 50.55 | 59.40 | 49.01 |

`m2m100_1.2b` and `seamless` (§9.6) both landed within their first
attempt at the pre-emptive 1e-4 default, no correction needed --
`m2m100_1.2b` is the new #2 overall and holds the best raw BLEU and TER
in the entire sweep.

**mt5_large is the new overall leader on GlossF1** (78.51) and chrF/
chrF++/TER, narrowly ahead of mbart50 (still best raw BLEU) and
essentially tied with afriteva_v2_large and nllb -- the top four are now
within about 2 GlossF1 points of each other and effectively
indistinguishable as a practical ranking, whatever the table order
implies.

This result **narrows an open question from the previous version of
this section.** `mt5` (base, 580M) and `afrimt5` (580M) were both
corrected to the same 1e-4-class learning rate as the top performers
(§5) and still trail by 15–20 GlossF1 points, which had been read as
"correction is necessary but not sufficient -- something about the
mT5 family specifically caps it." `mt5_large` (1.2B, same mT5 family,
same augmented data, same glossary metric) breaking into a tied-first
position complicates that reading: it suggests **model capacity**, not
the mT5 architecture or pretraining objective per se, is the more
likely explanation for why the base-size mT5 models plateau lower --
afriteva_v2_large (1B) and mt5_large (1.2B) are the two largest models
completed so far, and both are in the top four. This is still not
proven (confounded with the fact that the 1B+ tier also uses Adafactor
at a different learning rate, not a controlled capacity-only
comparison), but it is a more specific, testable hypothesis than before
`mt5_large` finished. Per-subject glossary breakdown for every
completed model is retained in each
`output_<model>/glossary_report.json` under `glossary.by_subject`.

### 7.1 m2m100 -- invalidated, corrected, re-run confirmed clean

The first m2m100 run completed all 6,560 steps cleanly (two infrastructure
bugs fixed along the way, §3.9–§3.10) and initially looked like the best
model in the sweep -- GlossAcc 73.18. That result did not hold up: it was
generated with a silently wrong language-control token (§3.11), which
leaked the literal text `"en "` into 97.0% of predictions. Its (invalid)
numbers were BLEU 65.85 / chrF++ 77.52 / GlossAcc 73.18 -- kept for the
record only, at `results_baseline_wrong_lang_token/m2m100/` and
`baseline_runs/output_m2m100/`.

The corrected re-run (§3.11's fix) completed cleanly. Verified directly:
0/2,623 predictions now contain the leaked token, and the current
numbers are close to, not wildly different from, the invalid run's --
**BLEU 66.38, chrF++ 78.66, TER 33.45, GlossAcc 73.15, GlossF1 73.28**
(table, §7). This is a useful data point in its own right: it suggests
the earlier "self-consistent but not actually language-conditioned"
explanation (§3.11) was largely correct -- the model was already
learning the task well from the parallel data regardless of the broken
language token, which is *why* the bug was easy to miss from the
numbers alone and had to be caught by reading actual predictions.

### 7.2 nllb -- learning-rate hypothesis confirmed

`nllb`'s first run (learning rate 1e-5, AdamW) completed its full
6,560-step schedule and converged normally -- `eval_loss` plateaued at
0.985–0.999 across its last five evaluations, not still descending --
yet still produced almost no glosses: GlossAcc 0.26, 78/2,726 glosses
produced. Unlike the original mt5 failure (early-stopped before
convergence, §5), this was a genuine, stable local optimum that simply
excluded glossing, which is why raising the learning rate was flagged
as an **unconfirmed hypothesis** rather than a repeat of a diagnosed
fix -- a different architecture reaching a stable bad optimum is not
obviously the same problem as one that never finished training.

The hypothesis is now confirmed. Re-run at 1e-4 (patience unchanged):

| | 1e-5 (original) | 1e-4 (re-run) |
|---|---|---|
| BLEU (clean) | 60.32 | **64.61** |
| chrF++ (clean) | 76.53 | **79.60** |
| GlossAcc | 0.26 | **79.86** |
| Glosses produced / expected | 78 / 2,726 | 2,196 / 2,726 |

Every axis improved simultaneously from a single hyperparameter change
-- the same "no tradeoff" signature as the original mt5/afrimt5 fix
(§5.4), now confirmed on a second, non-T5 architecture. The original
1e-5 run's full artifacts remain at `results_baseline_lr1e5/nllb/` and
`baseline_runs/output_nllb/` for the comparison. This strengthened, but
did not prove, that the same fix would help `mbart50` (§7.3) and the
four still-unstarted 1B+ T5-family models (§5.4) -- each remains its
own architecture and its own empirical question.

### 7.3 mbart50 -- first run, pre-emptive fix, current overall leader

`mbart50` had no prior run and no bug of its own to fix -- it went
straight to the corrected configuration (learning rate 1e-4, decided
pre-emptively on nllb's evidence in §7.2, before mbart50 had ever been
run) and completed its full 6,560-step schedule cleanly on the first
attempt. Verified no repeat of the m2m100-style token leak: 0/2,623
predictions contain a stray language-code token.

Result: **BLEU 68.18, chrF++ 79.96, TER 30.82, GlossAcc 75.90, GlossF1
76.74** -- the best BLEU, chrF, chrF++, TER, GlossPrecision, and GlossF1
of any model in the sweep so far (table, §7). Because this model never
had a "before," it cannot demonstrate the fix's effect the way nllb's
before/after comparison does -- but its outright leaderboard position is
itself evidence that pre-emptively applying a lesson learned from one
architecture to a different, untested one (§6's original caveat: "not
yet empirically confirmed... may not transfer") can pay off before
wasting a full run at a worse setting. Combined with nllb, this is now
two of two non-T5, non-Adafactor architectures where the 1e-5 → 1e-4
correction helped; it remains unknown whether it will hold for the
1B+ Adafactor tier (§5.4), which is a different optimizer entirely.

### 7.4 Compute cost

Wall-clock training time for the full 6,560-step schedule, 2× V100-32GB,
DDP (excludes data download/tokenization and the final beam-5 test
pass, which is separately timed in each `metrics.json`'s
`test_runtime`, typically 4–9 minutes):

| Model | Params | Optimizer | Batch/GPU | Training time | s/it |
|---|---|---|---|---|---|
| afriteva | 229M | adamw_torch | 8 | 1h 26m | 0.79 (1.26 it/s) |
| mbart50 | 680M | adamw_torch | 4 | 2h 18m | 1.27 |
| mt5 | 580M | adamw_torch | 4 | 2h 40m | 1.46 |
| afrimt5 | 580M | adamw_torch | 4 | 2h 41m | 1.47 |
| m2m100 | 418M | adamw_torch | 8 | 3h 03m | 1.67 |
| nllb | 600M | adamw_torch | 4 | 3h 44m | 2.05 |
| afriteva_v2_large | 1B | **adafactor** | **2** | 7h 09m | 3.92 |
| mt5_large | 1.2B | **adafactor** | **2** | **8h 11m** | **4.49** |
| toucan (2nd, successful run) | 1.2B | **adafactor** | **2** | 7h 26m | 4.08 |
| cheetah (2nd, corrected run) | 1.2B | **adafactor** | **2** | 7h 45m | 4.25 |
| seamless | 1.37B (text-only) | **adafactor** | **2** | 6h 51m | 3.76 |
| m2m100_1.2b | 1.24B | **adafactor** | **2** | 5h 52m | 3.22 |

Training time does not track parameter count monotonically among the
AdamW models (m2m100 at 418M took longer per step than mt5/afrimt5 at
580M) -- architecture and generation-related overhead during evaluation
(beam width, `gen_len`, vocabulary size) dominates over raw parameter
count at this scale. The jump to the 1B+/Adafactor tier is real and
large, though: both 1B+ models so far ran at roughly 2.5–3× the
per-step time of any AdamW model, driven by both the smaller batch size
(2 vs. 4–8, forced by memory, §6) and Adafactor's own overhead, and
scaling roughly with parameter count within this tier (1B → 3.92 s/it,
1.2B → 4.49 s/it). `eval_runtime` per evaluation pass was also several
times longer here than the AdamW-tier models -- evaluation alone cost
well over an hour of each of these runs' totals. `toucan`'s failed
first attempt (§3.17) ran at a similar per-step rate before being
stopped partway through; `cheetah`, still queued, should be expected
in the same multi-hour range.

### 7.5 afriteva_v2_large -- first 1B+/Adafactor run, pre-emptive fix confirmed a second time

The first model at the pre-emptively raised 1B+ tier rate (Adafactor,
1e-5 → 1e-3, §5.4/§10 -- flagged as "not yet empirically confirmed... may
not transfer" at the time it was set). It is now confirmed on this
architecture too: the run trained cleanly for its full 6,560 steps in
7h09m (batch size 2, far slower per step than any AdamW model, §7.4),
with an initially alarming but ultimately benign loss trajectory (23.1
→ 9.9 → 8.8 → 8.9 → 0.88 by the first evaluation -- high early loss
under Adafactor at this rate is evidently normal warm-up, not
divergence, at least for this model).

Final: **BLEU 66.32, chrF++ 79.57, TER 32.85, GlossAcc 78.58, GlossF1
76.54** -- essentially tied for second place overall (table, §7), and
zero leaked-token artifacts (verified, same check as §3.11/§7.1).

One caveat worth recording precisely: `eval_gen_len` stayed flat at
35–42 tokens for this model's *entire* training run -- the same signature
that, for nllb's original failed run, indicated no glossing was
happening at all (§7.2). Here it did not mean that; the final GlossAcc
is the second-highest in the sweep. The tokenizer for this model
apparently encodes the same augmented text into fewer tokens than
mT5/mBART's tokenizers do, so `gen_len` is **not directly comparable
across architectures** and should not be used alone as an early signal
for whether a model is learning to gloss -- only within a single
model's own training curve, where its trend (flat vs. rising) is still
informative.

### 7.6 mt5_large -- second 1B+/Adafactor run, new overall leader on GlossF1

Second confirmation of the pre-emptive 1e-3 Adafactor rate (§5.4/§9),
on the largest AdamW-family-adjacent model completed so far (1.2B
parameters, same `"translate English to Pidgin: "` prefix and
`google/mt5-large` checkpoint as its 580M sibling `mt5`). Trained
cleanly for the full 6,560 steps in 8h11m (4.49 s/it -- the slowest of
any model completed so far, §7.4), no errors, loss falling
smoothly and monotonically-ish from 0.816 (step 500) to 0.415 (step
3500) with no instability.

Final: **BLEU 66.92, chrF 82.16, chrF++ 80.59, TER 32.23, GlossAcc
79.68, GlossF1 78.51** -- the best GlossF1, chrF, chrF++, and TER in the
sweep so far (table, §7), narrowly ahead of mbart50 and
afriteva_v2_large. No leaked-token artifacts (same check as
§3.11/§7.1). See §7's revised discussion above for what this result
implies about model capacity vs. architecture as the explanation for
why the base-size mT5 models (`mt5`, `afrimt5`) plateau lower even
after the same learning-rate correction.

This is the third of four 1B+/Adafactor models to confirm the
pre-emptive fix (after afriteva_v2_large, §7.5); the fourth,
`toucan`, needed a real correction first (§3.17) -- see §7.7.

### 7.7 toucan -- required its own fix, then completed successfully (with an unexplained interruption along the way)

`toucan`'s first run at the sweep-wide 1e-3 Adafactor rate failed
outright (§3.17): eval BLEU stuck under 1.0 through 65% of training.
Diagnosis pointed at two candidates -- a source prefix that fragmented
into meaningless tokens, and 1e-3 possibly being too aggressive for
this checkpoint. Both were corrected together (prefix →
`"translate English to Pidgin: "`, LR → 1e-4) and the fix was confirmed
at the very first evaluation of the re-run: BLEU 38.82 vs. the failed
run's 0.06 at the identical step.

That corrected run was then interrupted by an external SIGTERM at 66%
progress, for reasons never identified (§3.18) -- not caused by
anything in this pipeline. A fresh run started automatically about a
minute later (not `--resume`, so the 66% of progress was lost, costing
~5h41m of redundant compute) and completed normally on its own.

Final result: **BLEU 64.74, chrF 81.04, chrF++ 79.34, TER 35.70,
GlossAcc 75.35, GlossF1 73.96** -- solidly mid-pack among the 1B+ tier
(table, §7), not the leader, but a legitimate, working flagship model,
which is what mattered most given how the first attempt looked.
Verified clean: 0 leaked-token artifacts in `test_predictions.csv`
(same check as §3.11/§7.1).

The subsequent `cheetah` run (§3.19), which started at the *original*
1e-3 with an *already-correct* prefix and failed identically to
`toucan`'s first attempt, is strong retrospective evidence that
**learning rate, not the prefix, was the actual cause of both
failures** -- the prefix fix likely helped `toucan` for unrelated
reasons (or not much at all), and the real fix, for both UBC-NLP
checkpoints specifically, was pulling Adafactor back from 1e-3 to
1e-4.

### 7.8 cheetah -- confirmed the LR fix, last model in the sweep, benchmark complete

`cheetah`'s corrected run (1e-4, §3.19) trained cleanly through all
6,560 steps in 7h45m with no further incidents -- no crash, no
external interruption this time, convergence continuing smoothly past
the step-2000 curve already recorded in §3.19.

Final: **BLEU 66.16, chrF 81.28, chrF++ 79.62, TER 33.80, GlossAcc
74.14, GlossF1 73.16** -- essentially tied with `toucan` (73.96) and
`m2m100` (73.28), landing squarely mid-pack once corrected, exactly as
expected for a model that needed a hyperparameter fix rather than
being fundamentally stronger or weaker than its peers. Verified clean:
0 leaked-token artifacts in `test_predictions.csv`.

**This closes the originally-planned 10-model sweep.** Every model
that was going to run has now run, with a valid, verified
`glossary_report.json`. Final standings, full table in §7:

1. mt5_large -- 78.51 GlossF1 (also best chrF, chrF++, TER)
2. mbart50 -- 76.74 (best BLEU, GlossPrecision)
3. afriteva_v2_large -- 76.54
4. nllb -- 75.88 (best GlossAcc, GlossPresence)
5. toucan -- 73.96
6. m2m100 -- 73.28
7. cheetah -- 73.16
8. afrimt5 -- 60.32
9. mt5 -- 56.11
10. afriteva -- 49.01

Five of the top seven models needed a real correction (bug fix or
learning-rate change) to reach these numbers. This was the 10-model
milestone; the roster subsequently grew to 12 (§9), and AfriCOMET
(§10) has since been implemented and scored against all of them --
see §12 for what remains open.

---

## 8. Artifacts and reproducibility

| File | Purpose |
|---|---|
| `prepare_data.py` | Rebuilds train/dev/test.json with glossary metadata from the source CSV |
| `train.py` | Single-model training entry point (`torchrun`-compatible, DDP) |
| `run_train.sh` | Launch wrapper: GPU auto-detection, CPATH fix (§3.7), clean process-group shutdown on interrupt |
| `glossary_metrics.py` | Glossary-accuracy metric + dual-reference MT scoring, importable and CLI |
| `evaluate_model.py` | Standalone re-scoring of a saved model without retraining (§3.5) |
| `run_benchmark.sh` | Full sweep runner -- skips already-completed models, continues past a failed one |
| `aggregate_results.py` | Collects all `output_*/glossary_report.json` into one leaderboard |
| `rerun_mt5_pair.sh` | The §5 re-run script (kept for the record / reproducibility) |
| `results_baseline_lr3e5/` | Pre-fix mt5/afrimt5 metrics only (§5) -- full weights in `baseline_runs/output_{mt5,afrimt5}/` |
| `results_baseline_lr1e5/nllb/` | Pre-fix nllb metrics only (§7.2) -- full weights in `baseline_runs/output_nllb/` |
| `results_baseline_wrong_lang_token/m2m100/` | Invalidated first m2m100 run's metrics (§3.11/§7.1) -- full weights in `baseline_runs/output_m2m100/` |
| `baseline_runs/` | Full model weights + checkpoints for every superseded run above |
| `benchmark_results.csv` / `.md` | Current leaderboard (regenerate with `aggregate_results.py`) |
| `BENCHMARK_REPORT.md` | This document |

Every run seeds Python/NumPy/PyTorch at 42; the dev-subsample used during
training is a seeded shuffle; the CSV→split recovery is deterministic.
Given the same code and data, all reported numbers should reproduce
exactly on the same hardware/library versions, and closely (MT metric
variation from beam-search/library nondeterminism, not model identity)
on different ones.

---

## 9. Model roster revision: paradigm diversity

After the original 10-model sweep completed (§7.8), the plan for the
remaining 4 slots changed. The original list (Pidgin-UNMT 220M, mT5-XL
3.7B, Cheetah-3.7B, TranslateGemma-4B) was dropped in favor of four
models chosen for architectural and pretraining-paradigm diversity
rather than more checkpoints from families already represented:
**MADLAD-400 3B**, **MADLAD-400 7B**, **T5 v1.1-XL**, and
**SeamlessM4T v2 Large**. The rationale: comparing only checkpoints
from the same handful of families (multiple mT5 variants, multiple
UBC-NLP models) says less about what generalizes than comparing across
genuinely different pretraining objectives and architectures --
including ones never exposed to Pidgin or even to translation as a
task, to observe how much fine-tuning alone can teach.

### 9.1 Pidgin-UNMT: investigated, then dropped rather than solved

Before the roster change, Pidgin-UNMT (`keleog/PidginUNMT`) was
investigated directly rather than assumed unworkable. Findings:

- It is a self-contained fork of Facebook's 2018 Unsupervised NMT
  research code (Lample et al.) -- a from-scratch Transformer with its
  own trainer, custom binarized data format, on-the-fly
  back-translation via multiprocess workers, and an adversarial
  discriminator. Nothing to do with `transformers`; a completely
  separate training system from the rest of this pipeline.
- A Colab notebook provided as a reference for fine-tuning it turned
  out to clone a *different* repository (`shashacode/PidginUNMT`, a
  fork) than the one it was meant to demonstrate. Diffing the two
  `main.py` argument lists directly confirmed this: the notebook's
  `--load_pretrained` flag exists only in that fork. The actual
  upstream repo uses `--reload_model` / `--reload_enc` / `--reload_dec`
  / `--reload_dis` instead, and the notebook's manual checkpoint
  "reformatting" step was traced through `src/model/seq2seq.py` and
  `src/utils.py` to be unnecessary -- and to produce the wrong key
  structure for either upstream reload path, versus the trainer's own
  native `save_model()` format, which loads directly.
- **The blocking issue**: the repo's `pretrained/README.md` links a
  Google Drive folder as the source for the trained model the paper
  reports (BLEU 7.93 pd->en, 5.18 en->pd). Checked three ways --
  `gdown` folder listing, a skip-download probe, and a raw HTTP fetch
  of the folder page. The folder loads (200 OK, not deleted or
  private) but contains zero files. The notebook's fork was also
  checked directly on GitHub for a committed checkpoint; none found.
  The pretrained model is not currently obtainable from any source
  available to this project.
- Also worth recording: the repo is licensed **CC BY-NC 4.0
  (NonCommercial)**, which would have constrained any release of a
  derivative regardless of the checkpoint issue.

No code was written against this broken premise. The roster change
above replaces it rather than pursuing a from-scratch unsupervised
pretraining run (the paper reports ~3 days on a V100 for that alone,
before any fine-tuning), which is a fundamentally larger undertaking
than "fine-tune a pretrained checkpoint."

### 9.2 Compatibility findings for the four replacements

All four were checked directly (config, tokenizer, license, gating)
before any training code was written, continuing the practice
established throughout this project of verifying rather than assuming
compatibility with a new checkpoint.

| Model | Params | HF class | Gated | pcm in vocab? |
|---|---|---|---|---|
| madlad400-3b-mt | ~2.9B | `T5ForConditionalGeneration` | No | No (493 tags checked; none) |
| madlad400-7b-mt | ~5.9B (33GB fp32) | `T5ForConditionalGeneration` | No | No (shares madlad-3b's tokenizer) |
| t5-v1_1-xl | ~2.85B | `T5ForConditionalGeneration` | No | N/A -- monolingual English, no language-tag system at all |
| seamless-m4t-v2-large | 1.37B (text-only) | `SeamlessM4Tv2Model`, needs `SeamlessM4Tv2ForTextToText` specifically | No | No (102 tags checked; has Igbo/Yoruba, no creole) |

MADLAD conditions via a **source-side language tag** (`<2kri> `, a
prefix, like the existing plain-prefix mechanism), not a decoder-side
`forced_bos_token_id` the way NLLB/mBART/M2M-100 do. Verified directly,
learning from the toucan incident (§3.16/§3.17): `<2kri>` tokenizes to
a single real vocab entry (id 300), not fragmented BPE pieces. Krio
(Sierra Leone Creole) was chosen over the more commonly-used Tok Pisin
proxy because it is the closest linguistic relative to Nigerian Pidgin
of any tag available in MADLAD's 493-language list -- both descend
from the same West African Pidgin English continuum. SeamlessM4T has
no creole option at all in its list, so it falls back to English
(`__eng__`), the same fallback rule already used for mBART-50/M2M-100
-- a genuine confound, stated as such, not a detail.

**SeamlessM4T needed real integration work, not just a config entry.**
`SeamlessM4Tv2Model` (the class `AutoModelForSeq2SeqLM` would resolve
to) is a full multimodal speech+text model; the dedicated
`SeamlessM4Tv2ForTextToText` class was required instead. Verified
directly before writing any training code:

- It loads **only** the text encoder/decoder weights (1.37B params,
  5.5GB in fp32), discarding the speech encoder, vocoder, and T2U
  components as "unexpected" -- confirmed against the repo's full
  9.2GB multimodal checkpoint size.
- A genuine forward+backward pass with `labels` runs cleanly (finite
  loss, no error) -- real `Seq2SeqTrainer` compatibility, not just
  inference.
- `tgt_lang` measurably changes label tokenization (checked: encoding
  identical text with `tokenizer.tgt_lang` set to two different
  languages produces two different token sequences), so `tokenizer.
  tgt_lang` must be set before encoding labels, mirroring the existing
  NLLB/mBART/M2M-100 pattern.
- Despite `generate()` exposing a `tgt_lang=` keyword the Trainer's own
  internal eval/predict generation calls have no way to supply, setting
  only `model.generation_config.forced_bos_token_id` and calling
  `generate()` with no `tgt_lang` still produces a first generated
  token exactly matching that id, and genuinely-decoded text in the
  target language (tested by forcing Yoruba as a control case) -- so
  the existing forced-bos mechanism transfers without needing any
  `Seq2SeqTrainer` subclassing.

### 9.3 madlad400-3b-mt and t5-v1_1-xl: hit a genuine memory ceiling, deferred to LoRA

Both were smoke-tested successfully (tokenizer, model load, real
`generate()` call, no crash) before launching full training -- but both
failed immediately at the start of actual training with
`CUDA out of memory`, failing on a **20MB** allocation with the GPU
already at 31.23/31.73GB used.

Diagnosis, in order: sequence length was ruled out first -- madlad3b
was re-launched at `max_source_length`/`max_target_length` 128 (versus
the default 200; checked directly that this loses under 1% of training
examples) specifically to test this, and it produced the **identical**
OOM at the **identical** 31.23GB, meaning sequence length was never the
driver. The actual cause: both checkpoints' loader logs show input and
output embeddings are genuinely **untied** ("both are present in the
checkpoints with different values, so we will NOT tie them"), meaning
each stores its large vocabulary embedding table twice rather than
once. Combined with plain fp32 weights (~11.8GB) and fp32 gradients
(~11.8GB) under this pipeline's DDP-full-replica-per-GPU design, this
leaves essentially zero headroom on a 32GB GPU before any activation
memory at all -- a fixed-footprint ceiling, not a tunable one.

Given the choice between building FSDP/DeepSpeed ZeRO support (real new
infrastructure, with generation-under-sharding being a known source of
subtle bugs that would need real validation before trusting a
multi-hour run) versus deferring these two to the PEFT/LoRA phase
(where trainable-parameter memory drops by roughly two orders of
magnitude and this ceiling should not apply at all), the latter was
chosen. Both configs are preserved, commented, in `train.py`'s excluded
block -- including the Krio proxy-token research for madlad3b -- so
this is a deferral, not lost work.

### 9.4 madlad400-7b-mt replaced with m2m100_1.2B

madlad400-7b-mt's exclusion (§9.3-adjacent, confirmed infeasible
outright at 33GB fp32 weights alone) was replaced with
`facebook/m2m100_1.2B` rather than left as a gap in the roster. Same
`M2M100ForConditionalGeneration` class and target-language mechanism as
the already-completed 418M variant (§3.9, §3.12) -- no new integration
work needed, just a new `MODEL_CONFIGS` entry and a pre-flight
`generate()` check (confirmed clean before queueing: correct bos id
128022, matching the 418M variant's verified value, no crash). The two
`m2m100-12B-*` checkpoints were considered and ruled out directly:
d_model=4096, roughly 1.7x the size of the madlad-7b that already
OOM'd, clearly infeasible for full fine-tuning under this pipeline.

### 9.5 Research design: three-stage evaluation and a name for the flagship

The evaluation plan for this roster was formalized: every model is
scored **zero-shot**, after **full fine-tuning**, and (for models that
support it) after **LoRA/PEFT** fine-tuning, against the same metric
set (BLEU, chrF++, AfriCOMET -- unimplemented at the time this plan was
written, done as of §10 -- glossary-accuracy).
Zero-shot is deliberately sequenced *after* the full fine-tuning phase
completes, not interleaved with it, so it runs as one clean, systematic
pass across every model in the final roster rather than being repeated
piecemeal in a leaked-token-check-only spirit like the informal
pre-flight `generate()` checks used throughout this project. This
answers six explicit research questions, from "how capable are
existing multilingual models before any Pidgin exposure" (zero-shot
alone) through "which pretrained model is most adaptable" (fine-tuning
gain, zero-shot vs. fine-tuned) to "can PEFT match full fine-tuning at
a fraction of the trainable parameters" (full FT vs. LoRA).

**"PidginEdu-LLM" is not a separate model to be built** -- it is
whichever model in the completed fine-tuned roster scores best
(glossary-augmented, evaluated against the full metric set), decided
empirically once fine-tuning is done, not assumed in advance.

**`mt5_large` is the automated-metric leader among the 12
full-fine-tuned models** -- leads 3 of 4 metrics: **GlossF1** (78.51,
the benchmark's headline metric given its glossary-augmented framing),
**AfriCOMET** (71.94), and **chrF++** (80.59) -- and is a close third
on raw BLEU (66.92, versus `m2m100_1.2b`'s leading 68.50, a 1.6-point
gap). No other model leads more than one metric. `madlad3b` and
`t5-v1_1-xl` are not eligible candidates under the metrics-only rule
below, since neither was ever fully fine-tuned (§9.3/§11.3,
memory-ceiling excluded) -- their only result is LoRA.

All preconditions for a metrics-only finalization were met -- the
full-fine-tuning phase is complete (12 models, §7/§9), the zero-shot
phase is complete (§9.7), and the LoRA/PEFT phase is complete (§11)
but was excluded from competing for this designation on the grounds
that full fine-tuning represents the best achievable quality for a
given checkpoint while LoRA answers a separate question (RQ4,
cost/adaptability). Human evaluation, listed as a candidate metric in
early planning, was explicitly decided against as a *deciding
criterion* at that point in the project -- automated metrics alone
were meant to settle it.

**Selection methodology correction, and independent confirmation.**
The comparison above ranks models by their **test-set** scores. Doing
that alone to *choose* a winner among several candidates is a real
methodological problem, not just a style choice: picking whichever of
12 models happens to score highest on the same held-out set that will
also be reported as its final quality number is a multiple-comparison/
winner's-curse pattern, and can overstate the selected model's true
generalization performance. The methodologically correct order is
*validation performance decides the winner, test performance reports
its quality* -- caught during an external review of this benchmark's
methodology and treated as a real gap to close, not a formality.

Built a matching validation-set leaderboard: each of the 12 models'
already-saved best-by-validation-chrF checkpoint (§6's
`metric_for_best_model="chrf"`, `load_best_model_at_end=True` --
selection *within* a model's own training was already validation-based
and did not need correcting) was re-scored against the **dev split**
(2,623 examples, held out from training and distinct from test) at the
same beam-5 settings used for every test-set number in this report.
This surfaced a serious, unrelated bug along the way -- §3.20, 6 of the
12 models' saved checkpoints reloaded as fluent-looking garbage due to
a silently corrupted `tie_word_embeddings` flag, `mt5_large` included
-- fixed there (no retraining required) before trusting any of these
numbers.

| Model | Dev BLEU | Dev chrF++ | Dev GlossAcc | Dev GlossF1 |
|---|---|---|---|---|
| mt5_large | 66.51 | **79.65** | 79.58 | **78.50** |
| m2m100_1.2b | 54.27 | 78.01 | 76.56 | 76.77 |
| afriteva_v2_large | 65.95 | 79.27 | 77.53 | 76.26 |
| nllb | 63.47 | 78.73 | **80.03** | 75.99 |
| mbart50 | **67.30** | 79.27 | 74.73 | 75.66 |
| seamless | 67.17 | 79.59 | 75.51 | 75.18 |
| m2m100 | 53.75 | 77.25 | 74.13 | 74.15 |
| toucan | 64.25 | 78.28 | 75.18 | 73.96 |
| cheetah | 64.91 | 78.48 | 74.58 | 73.84 |
| afrimt5 | 61.61 | 77.24 | 54.76 | 59.49 |
| mt5 | 60.18 | 76.57 | 48.71 | 55.17 |
| afriteva | 55.40 | 73.01 | 42.18 | 49.42 |

**`mt5_large` wins on validation too** -- leads dev GlossF1 (78.50,
essentially identical to its own test-set 78.51) and dev chrF++
(79.65), the same two metrics it led on test, with `mbart50` leading
dev BLEU just as `m2m100_1.2b` led test BLEU. The ranking is stable
across both splits, not an artifact of which one happened to be
scored: **`mt5_large` is confirmed the automated-metric leader under
the correct selection methodology, not just the original (flawed)
one.**

**Flagship override: `toucan` selected as PidginEdu-LLM instead of the
automated-metric leader.** After the metrics-only process above was
completed and validated, the project author reviewed generated
translations from both `toucan` and `mt5_large` directly and judged
`toucan`'s output to read as more natural, more fluent Nigerian Pidgin
-- despite scoring lower on every automated metric in this benchmark.
On that basis, **`toucan` is the finalized PidginEdu-LLM**, not
`mt5_large`.

This is stated plainly as what it is: a qualitative, single-reviewer
judgment call by the project author, made *after* and *in spite of* a
metrics-only process that had already been carefully built specifically
to avoid exactly this kind of ad hoc override (§9.5's original
framing, the validation-set correction earlier in this section). It is
not a new evaluation protocol -- no sample size, no blind rating, no
second rater, no defined rubric -- and should not be read as one. It
is disclosed here rather than silently substituted, because the
alternative (quietly replacing the metrics-selected model with the
author-preferred one and leaving the metrics section reading as if it
had produced this result) would misrepresent how the decision was
actually made. Automated metrics and human perception of translation
quality are well known in MT research to sometimes diverge; a rigorous
resolution of that divergence would be a real, structured human
evaluation study (sample of outputs, multiple blind raters, an
inter-rater agreement figure) -- exactly the kind of study this
benchmark decided at the outset not to build (§12 item 8's original
form). This override does not retroactively become that study by
being written down; it is one person's read of a handful of outputs,
presented honestly as such.

`mt5_large` remains fully documented in this report and its checkpoint
remains a real, valid result of this benchmark -- the automated-metric
leader among 12 fully fine-tuned models, confirmed on both test and
validation splits. It was removed from public hosting at the project
author's explicit request, a decision this document records but does
not endorse or dispute on technical grounds -- there is no metric-based
finding in this benchmark that argues for removing it.

### 9.6 seamless and m2m100_1.2B: both completed, both landed strong

Both finished overnight, unattended, with no errors -- `seamless` in
6h51m, `m2m100_1.2b` in 5h52m. Verified clean (no leaked-token
artifacts, same check as §3.11/§7.1) before trusting either result.

| Model | BLEU | chrF++ | GlossAcc | GlossF1 |
|---|---|---|---|---|
| m2m100_1.2b | **68.50** | 80.14 | 77.15 | 77.20 |
| seamless | 67.56 | 79.74 | 75.75 | 75.52 |

`m2m100_1.2b` is now **#2 overall** by GlossF1 (best raw BLEU in the
entire sweep) and `seamless` lands solidly mid-upper-pack, ahead of
`toucan`, `m2m100` (418M), and `cheetah`. This is real evidence for the
paradigm-diversity rationale (§9): both are architecturally distinct
from the mT5-family models that otherwise dominate the top of the
leaderboard, and neither needed a learning-rate correction the way
`toucan`/`cheetah` did -- the pre-emptive 1e-4 default (established
from that earlier failure, §5.4) held on the first attempt for both.

**This closes the full-fine-tuning phase of the revised roster.**
12 of the eventual 14 models are done (`madlad3b`/`t5-v1_1-xl` remain
deferred to the LoRA phase, §9.3). The zero-shot sweep (§9.5) is next.

### 9.7 Zero-shot sweep: all 14 models scored, no correction needed

Building the sweep required a real rewrite of `evaluate_model.py`
first -- the original version assumed a *local fine-tuned* checkpoint
directory, used plain `AutoModelForSeq2SeqLM` unconditionally (would
have crashed on toucan/cheetah's tokenizer bug, §3.16, and loaded the
wrong class entirely for seamless, §9.2), had no language-token
handling for the five models that need it (§3.12/§9.2), and -- most
importantly -- **wrote results directly into the model's own output
directory**, which for zero-shot would have silently overwritten the
already-computed fine-tuned results. Rewritten to mirror train.py's
per-model loading logic (duplicated by necessity, not import -- train.py
is a top-level script with side effects on import, not a library;
noted in-code as needing manual sync if train.py's handling changes),
support loading either a local checkpoint or the original HF
checkpoint, and write zero-shot output to a separate
`output_zeroshot_<key>/` directory.

**One new bug found and fixed during verification**: the custom
`PreTrainedTokenizerFast` built for toucan/cheetah (§3.16) emits
`token_type_ids`, which T5-style `generate()` rejects outright. This
never surfaced during actual training because `DataCollatorForSeq2Seq`/
`Trainer` filter model inputs; this script's manual generation loop did
not. Fixed by explicitly passing only `input_ids`/`attention_mask` to
`generate()` rather than assuming tokenizer output is already exactly
right. Caught before the full sweep, not after.

**A second, unrelated bug found while verifying the re-score path**:
`AutoTokenizer.from_pretrained()` on *any* of the fine-tuned models'
saved directories failed with `AttributeError: 'list' object has no
attribute 'keys'`, tracing to a `tokenizer_config.json` field
(`extra_special_tokens`) that the `transformers` version active when
these checkpoints were saved wrote as a list, but the currently
installed version expects as a dict. Confirmed this field does not
exist at all in the original upstream checkpoints -- it is a
convenience mirror of the actual vocabulary (which lives untouched in
`tokenizer.json`), not load-bearing. Patched by removing the key from
9 of 12 saved `tokenizer_config.json` files (the other 3 -- afriteva,
cheetah, toucan -- never had it, for reasons tied to how each was
originally saved); verified with a fresh load afterward for the
language-token models specifically, and confirmed the re-score path
now reproduces afriteva's already-published numbers exactly.

All 14 models -- the 12 fine-tuned plus `madlad3b`/`t5-v1_1-xl`
(zero-shot is unaffected by their training-memory ceiling, §9.3, since
there is no gradient or optimizer state involved) -- were then scored
with **greedy decoding (`num_beams=1`) uniformly throughout**, a
deliberate methodological choice distinct from the beam-5 used for
final fine-tuned test scores: this is a baseline-establishing pass, not
a final polished number, and standardizing on the cheaper setting kept
14 models (two of them ~2.9B and never run through this pipeline
before at all) tractable. `afriteva`'s first zero-shot attempt used
beam-5 by oversight and was redone at beam-1 for consistency with the
rest of the table. Completed unattended over ~2h30m with zero errors
across all 14 runs.

| Model | Zero-shot BLEU | chrF++ | AfriCOMET |
|---|---|---|---|
| seamless | **32.71** | **62.53** | 58.55 |
| m2m100_1.2b | 28.23 | 58.85 | **59.42** |
| m2m100 | 27.08 | 57.93 | 58.33 |
| toucan | 17.01 | 40.68 | 50.47 |
| afriteva_v2_large | 5.84 | 19.42 | 21.74 |
| madlad3b | 5.80 | 25.96 | 32.16 |
| t5v11xl | 3.16 | 11.88 | 22.09 |
| mbart50 | 2.34 | 10.19 | 41.24 |
| nllb | 2.08 | 14.20 | 23.51 |
| mt5_large | 0.97 | 15.83 | 25.70 |
| afriteva | 0.30 | 4.82 | 16.42 |
| mt5 | 0.28 | 4.88 | 18.39 |
| afrimt5 | 0.03 | 4.31 | 21.84 |
| cheetah | 0.00 | 0.01 | 15.85 |

AfriCOMET broadly agrees with BLEU/chrF++ on the top and bottom of this
ranking (`m2m100_1.2b`/`seamless`/`m2m100` lead; `cheetah` trails), but
disagrees more than usual in the middle: `mbart50` (BLEU 2.34, rank 8)
scores AfriCOMET 41.24 (rank 5) -- notably higher than its n-gram-based
rank would suggest, and `toucan` similarly outranks several higher-BLEU
peers on AfriCOMET. Consistent with AfriCOMET being a semantic-adequacy
metric rather than a surface-overlap one (§10.3's observation about its
tighter clustering among the fine-tuned models applies to some degree
here too, though the zero-shot spread is far wider overall given how
much weaker most of these outputs are).

Glossary accuracy is 0 (or within noise of 0) for every model, as
expected -- none have ever seen a glossed target. This spread directly
answers RQ1 ("how capable are existing multilingual models before any
Pidgin exposure"): `seamless` and both `m2m100` variants retain real
translation capability zero-shot (most likely from broader multilingual
pretraining coverage that happens to include English-adjacent creoles
or simply more languages generally), while most T5-family checkpoints
-- `mt5`, `afrimt5`, `cheetah` -- are close to non-functional for this
pair without any Pidgin exposure at all.

**The fine-tuning-gain table (RQ2/RQ3) shows a striking, coherent
pattern**, computed for the 12 models with both a zero-shot and a
fine-tuned BLEU on file:

| Model | Zero-shot | Fine-tuned | Gain |
|---|---|---|---|
| cheetah | 0.00 | 66.16 | **+66.16** |
| mt5_large | 0.97 | 66.92 | +65.95 |
| mbart50 | 2.34 | 68.18 | +65.84 |
| afrimt5 | 0.03 | 63.41 | +63.38 |
| nllb | 2.08 | 64.61 | +62.53 |
| mt5 | 0.28 | 62.65 | +62.37 |
| afriteva_v2_large | 5.84 | 66.32 | +60.48 |
| afriteva | 0.30 | 57.71 | +57.41 |
| toucan | 17.01 | 64.74 | +47.73 |
| m2m100_1.2b | 28.23 | 68.50 | +40.26 |
| m2m100 | 27.08 | 66.38 | +39.31 |
| seamless | 32.71 | 67.56 | +34.85 |

**The models with the weakest zero-shot performance show the largest
fine-tuning gains, and all 12 converge toward a similar final BLEU band
(57.7-68.5) regardless of where they started** (`cheetah`: 0.00 -> 66.16;
`seamless`: already at 32.71 -> 67.56 -- a much smaller absolute jump to
reach a similar endpoint). This is close to an inverse-linear
relationship between zero-shot capability and fine-tuning gain, and is
arguably the single most paper-relevant finding produced by this
benchmark so far: it suggests Eng-PidginEdu's value is not concentrated
in "helping already-capable models get better" but in **closing an
architecture-independent gap to a common ceiling** -- direct, headline
evidence for the dataset's contribution (RQ2's framing: "arguably the
most important result for your dataset paper"), independent of which
pretrained checkpoint is used.

AfriCOMET scoring for the zero-shot set is complete (table above;
§10's methodology applies unchanged) -- all 14 models scored, no
failures. Zero-shot logs (per-model generation logs, the chain
orchestration log, and the AfriCOMET scoring log) are preserved in
`logs/zeroshot_sweep/` rather than only in ephemeral session-scoped
temp storage.

---

## 10. AfriCOMET metric

The fourth metric in the evaluation framework (§9.5's diagram: BLEU,
chrF++, AfriCOMET, glossary-accuracy) was unimplemented until this
point (§10-item-4, previously). Implemented as `africomet_metrics.py`,
scoring all 12 completed models' saved predictions directly -- no
retraining required, same principle as `evaluate_model.py`.

### 10.1 Checkpoint selection and verification

**`masakhane/africomet-stl-1.1`** (Wang et al., 2023,
arXiv:2311.09828) -- reference-based (needs source + hypothesis +
reference, not reference-free QE), Apache-2.0, not gated. Checked
directly before use, not assumed: its model card lists 76 supported
languages, and **`pcm` is explicitly among them** -- unlike almost
every other multilingual resource touched in this project (NLLB,
mBART-50, M2M-100, MADLAD, SeamlessM4T all lacked Nigerian Pidgin
entirely, §3.12/§9.2), this metric was actually built with Nigerian
Pidgin in scope. Built on an "Afro-XLM-R-large-76L" encoder,
specifically tuned for African-language MT evaluation and validated in
the WMT 2024 Metrics Shared Task per its model card.

Installed `unbabel-comet` (2.2.7) into the existing environment;
checked for dependency conflicts with the installed `pytorch-lightning`
(2.6.5) and `torchmetrics` (0.10.3) before trusting it, then verified
the full pipeline end-to-end on real project data (five real
predictions from `mt5_large`'s saved test output) before running it at
scale: model downloads and loads cleanly, `model.predict()` returns
per-segment scores in [0, 1] plus a system-level average, and the
first real batch produced a plausible score (system average 0.729)
rather than a placeholder or error.

### 10.2 Scoring methodology

Scored against **clean references** (glosses stripped from both
prediction and reference via `glossary_metrics.py`'s existing
`strip_glosses()`, reused rather than reimplemented) -- the same
clean/augmented dual-reference principle already established for
BLEU/chrF++ (§4.3), applied here for a different reason: AfriCOMET was
trained on natural parallel sentences, not text containing inline
parenthetical glosses, so scoring it against the glossary-augmented
form would push every example out of its training distribution for no
benefit -- gloss quality is already measured separately by the
glossary-accuracy metric (§4). Raw model output (0-1) is rescaled to
0-100 (`africomet_stl`) to sit on the same visual scale as
BLEU/chrF++/GlossF1 in the leaderboard; the untransformed value is
also kept (`africomet_stl_raw`) for anyone who wants it.

### 10.3 Results

| Model | AfriCOMET |
|---|---|
| mt5_large | **71.94** |
| seamless | 71.92 |
| cheetah | 71.84 |
| mbart50 | 71.80 |
| toucan | 71.80 |
| m2m100_1.2b | 71.76 |
| afriteva_v2_large | 71.62 |
| nllb | 71.50 |
| m2m100 | 71.26 |
| afrimt5 | 70.82 |
| mt5 | 70.63 |
| afriteva | 63.21 |

`mt5_large` leads on AfriCOMET too, one of the three automated metrics
it leads overall (§9.5) -- confirmed here across a fourth, independent
metric family (neural quality estimation rather than n-gram overlap or
the glossary-specific metric). `toucan`, the finalized PidginEdu-LLM
(§9.5, selected on qualitative grounds rather than automated metrics),
scores 71.80 here -- close behind, within the tight band described
next. One genuine
observation worth carrying into the paper: **AfriCOMET's spread across
competent models is far tighter than BLEU's or GlossF1's** -- eleven
of the twelve models fall within a 1.3-point band (70.6-71.9), while
the same eleven span nearly 20 GlossF1 points (56.1-78.5) and 6 BLEU
points. Only `afriteva` (the smallest, weakest model, already last on
every other metric) separates clearly, at 63.21. This is consistent
with neural QE-style metrics generally compressing differences between
systems that are all "reasonably fluent and adequate" more than
surface n-gram metrics do -- worth stating explicitly rather than
treating the tight clustering as AfriCOMET being uninformative; it is
measuring something real, just on a different, more saturated scale
for this quality range.

---

## 11. LoRA/PEFT fine-tuning (RQ4)

### 11.1 Implementation

Added to `train.py` rather than a separate script, so LoRA runs share
the exact same data pipeline, tokenization, language-token resolution,
and glossary/AfriCOMET scoring as every full fine-tune already in this
report -- only the model-wrapping and optimizer/learning-rate defaults
differ. `--lora` freezes the base model and wraps it with `peft`
(0.20.0)'s `get_peft_model()`; `--lora-r`/`--lora-alpha`/
`--lora-dropout`/`--lora-target-modules` expose the adapter
hyperparameters, defaulting to `r=16`, `alpha=32`, `dropout=0.05`.

Three defaults are forced specifically under `--lora`, independent of
the model's own full-fine-tune config:

- **Optimizer**: always `adamw_torch`, regardless of the model's
  `optim` entry. Adafactor's memory saving over AdamW (the reason the
  1B+ tier uses it for full fine-tuning, §6) is moot once the base
  model is frozen -- there is no optimizer state for ~98%+ of the
  parameters either way.
- **Learning rate**: `1e-3` (the original LoRA paper's starting
  point), well above every full-fine-tune rate in this sweep (max
  `1e-3` only for the Adafactor tier, `1e-4` or below otherwise).
  LoRA's randomly-initialized A / zero-initialized B matrices start
  further from a useful update than a pretrained full-fine-tune
  weight does, so it conventionally wants a higher rate.
- **Output directory**: `output_lora_<model>` instead of
  `output_<model>`, so a LoRA run can never silently overwrite an
  already-scored full fine-tune.

Verified in isolation before touching the real pipeline: `LoraConfig`
+ `get_peft_model()` on `castorini/afriteva_base` produced the
expected "1,769,472 trainable / 230,797,824 total (0.77%)" split;
`save_pretrained()` wrote only a 7.1MB adapter, not the 230M-param
base; and a real forward+backward pass under
`gradient_checkpointing_enable(gradient_checkpointing_kwargs=
{"use_reentrant": False})` confirmed the known PEFT+checkpointing
gotcha directly -- gradients only flow correctly if
`model.enable_input_require_grads()` is also called, which is now
wired in conditionally (`if training_args.gradient_checkpointing`)
right after `get_peft_model()` in `train.py`.

**A launcher bug found during the first end-to-end smoke test, unrelated
to LoRA itself**: running `train.py` as plain `python` on a single
pinned GPU (`CUDA_VISIBLE_DEVICES=0 python train.py ...`, the pattern
`run_train.sh` already used for single-GPU jobs) now hard-crashes with
`ValueError: Default process group has not been initialized` inside
`accelerate`'s `PartialState`. Traced to the installed `accelerate`
version unconditionally reaching a `torch.distributed.get_world_size()`
call once `self.backend` resolves to `"nccl"` (which it does whenever
CUDA is available, launched or not), while the `init_process_group()`
call earlier in the same function is gated behind
`LOCAL_RANK != -1` -- a variable only `torchrun` sets. Plain `python`
therefore reaches the `get_world_size()` call with no process group
ever created. Not specific to `--lora`; this would affect any
single-GPU run under the currently installed `transformers`/
`accelerate` versions. Worked around by always launching through
`torchrun --standalone --nnodes=1 --nproc_per_node=1` even for a single
pinned GPU, which sets `LOCAL_RANK=0` and takes the code down the path
that actually calls `init_process_group()`. All LoRA runs in this
report use this launch pattern; `run_train.sh`'s plain-`python`
single-GPU branch has not been fixed and would need the same change if
used again as-is.

**Full per-model LoRA configuration, exactly as run** (pulled from
each run's own logged header and PEFT's own trainable-parameter print,
not reconstructed from the config file after the fact):

| Model | Per-GPU batch | Grad accum | Effective batch | LR | Optimizer | `r` | `alpha` | `dropout` | `target_modules` | Trainable / total | Trainable % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| afriteva | 8 | 2 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q,v,wi,wo` | 4,718,592 / 233,746,944 | 2.02% |
| m2m100 | 8 | 2 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q_proj,v_proj,fc1,fc2` | 6,291,456 / 490,196,992 | 1.28% |
| mt5 | 4 | 4 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q,v,wi,wo` | 2,850,816 / 585,252,096 | 0.49% |
| afrimt5 | 4 | 4 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q,v,wi,wo` | 2,850,816 / 585,252,096 | 0.49% |
| nllb | 4 | 4 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q_proj,v_proj,fc1,fc2` | 6,291,456 / 621,365,248 | 1.01% |
| mbart50 | 4 | 4 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q_proj,v_proj,fc1,fc2` | 6,291,456 / 617,170,944 | 1.02% |
| afriteva_v2_large | 2 | 8 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q,v,wi,wo` | 7,667,712 / 1,032,514,560 | 0.74% |
| mt5_large | 2 | 8 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q,v,wi,wo` | 7,667,712 / 1,237,249,024 | 0.62% |
| toucan | 2 | 8 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q,v,wi,wo` | 7,667,712 / 1,237,249,024 | 0.62% |
| cheetah | 2 | 8 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q,v,wi,wo` | 7,667,712 / 1,237,249,024 | 0.62% |
| seamless | 2 | 8 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q_proj,v_proj,fc1,fc2` | 18,874,368 / 1,389,406,208 | 1.36% |
| m2m100_1.2b | 2 | 8 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q_proj,v_proj,fc1,fc2` | 18,874,368 / 1,258,344,448 | 1.50% |
| madlad3b | 4 | 4 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q,v` (pre-fix -- run before §11.2's bug was found) | 9,437,184 / 2,949,811,200 | 0.32% |
| t5v11xl | 4 | 4 | 16 | 1e-3 | adamw_torch | 16 | 32 | 0.05 | `q,v` (pre-fix -- run before §11.2's bug was found) | 9,437,184 / 2,859,194,368 | 0.33% |

Per-GPU batch/effective batch are each model's own full-FT config
value, reused unchanged for LoRA (§11.1's design choice was to force
LR/optimizer/output-dir under `--lora`, not batch size -- these
configs were already tuned to fit comfortably in the tighter full-FT
memory budget, so they were expected to fit LoRA's far lower memory
footprint with margin to spare, which held true in every run --
peak GPU memory stayed at 13-19GB out of 32GB even for the two largest
models, §11.3). `r`, `alpha`, `dropout`, and LR/optimizer were never
overridden from their CLI defaults for any run in this report.

### 11.2 A genuine methodological bug: attention-only target modules failed the glossary task

The first two models run under LoRA -- `madlad3b` and `t5-v1_1-xl`,
using the original LoRA paper's default `target_modules=['q','v']`
(T5Attention's projection names, confirmed via `named_modules()`) --
both completed cleanly and scored well (§11.3). Encouraged by that,
the same `q,v`-only config was extended to the rest of the roster. The
first pair run this way, `afriteva` and `seamless`, surfaced a real
problem: `afriteva`'s glossary accuracy came back at **0.04%** (1
correct gloss out of 2726 expected), against **41.71%** for its own
full fine-tune -- despite normal-looking loss and BLEU curves
throughout training (`test_bleu` 38.44, only moderately below the
full-FT run's 55.48).

Reading the actual generated text made the mechanism clear rather than
leaving it as a mysterious number: `term_mention_rate` stayed at 83.4%
(full-FT: 81.5%) -- the model still *names* the right terminology --
but `produced_glosses` collapsed to 47 against 1914 for the full
fine-tune. It kept the term, dropped the parenthetical explanation:

```
REF : ...acceptable (dem worthy of acceptance or satisfactory) bfoqs
PRED: ...acceptable bfoqs.
```

Only 448/2623 test predictions contained a `(` at all, against
1759/2623 in the references. `m2m100` and `seamless` (BART-style,
`q_proj`/`v_proj`) showed the same pattern at smaller scale --
`gloss_accuracy` dropped from 73.15%/75.75% (full-FT) to 33.82%/33.57%
under `q,v`-only LoRA, roughly 45% retention rather than `afriteva`'s
near-total collapse.

**Diagnosis**: attention-only LoRA reweights what the model attends
to; it does not touch the feed-forward projections (`wi`/`wo` for
T5-family, `fc1`/`fc2` for BART-style -- both confirmed present via
`named_modules()` across every affected checkpoint before the fix was
applied, not assumed by family resemblance). Inserting a specific,
correct gloss for a specific term is a content-injection task, closer
to memorizing new lexical associations than to steering fluency, and
that capability lives in the FFN layers -- which stayed frozen.
`afriteva` (229M, the smallest model in the roster) collapsed hardest
because it has the least latent pretrained capacity for a rank-16
attention adapter to redirect; a small model that never had much
Pidgin-terminology exposure needs real weight movement to learn the
glossary mappings from scratch, which is what full fine-tuning does
and attention-only LoRA does not.

**Fix**: extended `target_modules` to include the FFN projections --
`q,v,wi,wo` for T5-family checkpoints, `q_proj,v_proj,fc1,fc2` for
BART-style ones (`m2m100`, `m2m100_1.2b`, `mbart50`, `nllb`,
`seamless`) -- via a new per-model `lora_target_modules` config entry
in `MODEL_CONFIGS`, resolved as
`args.lora_target_modules or config.get("lora_target_modules", "q,v,wi,wo")`.
Trainable-parameter fraction rises modestly (e.g. `afriteva`:
0.77% -> 2.02%; `m2m100`: 0.48% -> 1.28%) but stays far below full
fine-tuning. `mt5_large`, which was 25% through its `q,v`-only run when
this was found, was killed and restarted from scratch under the fixed
config rather than left inconsistent with the rest of the sweep.
`afriteva`, `m2m100`, and `seamless` were likewise discarded and
re-run in full.

**Result after the fix, final -- all 12 re-run/remaining models complete**:

| Model | Full-FT GlossAcc | LoRA (`q,v` only) | LoRA (`q,v,wi,wo`/`+fc1,fc2`) | Retention |
|---|---|---|---|---|
| seamless | 75.75 | 33.57 | **78.36** | 103% (exceeds) |
| mbart50 | 75.90 | n/a (only run once, post-fix) | **74.36** | 97.9% |
| nllb | 79.86 | n/a (only run once, post-fix) | **77.55** | 97.1% |
| m2m100_1.2b | 77.15 | n/a (only run once, post-fix) | **73.66** | 95.5% |
| m2m100 | 73.15 | 33.82 | **69.85** | 95.5% |
| toucan | 75.35 | n/a (only run once, post-fix) | **62.47** | 82.9% |
| cheetah | 74.14 | n/a (only run once, post-fix) | **57.78** | 77.9% |
| afriteva_v2_large | 78.58 | n/a (only run once, post-fix) | **59.35** | 75.5% |
| mt5_large | 79.68 | n/a (killed mid-run, restarted) | **58.88** | 73.9% |
| afriteva | 41.71 | 0.04 | **24.17** | 57.9% |
| afrimt5 | 56.09 | n/a (only run once, post-fix) | **20.51** | 36.6% |
| mt5 | 50.11 | n/a (only run once, post-fix) | **11.99** | 23.9% |

(`madlad3b`/`t5-v1_1-xl` have no full-FT baseline to compute a
retention percentage against -- they were excluded from full
fine-tuning entirely by the memory ceiling in §9.3/§11.3; LoRA is
their only result, GlossAcc 73.07/66.14.)

**The fix worked, and the final 12-model picture is a clean,
consistent architecture split -- not noise, not a single-model
anomaly.** Every BART-style model (`seamless`, `mbart50`, `nllb`,
`m2m100`, `m2m100_1.2b`) retained 95-103% of its full-FT glossary
score; LoRA is essentially equivalent to full fine-tuning for this
entire family on this task. The 1B+ T5-family models (`toucan`,
`cheetah`, `afriteva_v2_large`, `mt5_large`) land in a consistent
middle band, 74-83% retention. The base-tier T5-family models
(`afriteva`, `afrimt5`, `mt5`, all under 600M) are the clear outliers,
retaining only 24-58% -- and within that group, worse for smaller: `mt5`
(580M) worst at 23.9%, `afrimt5` (580M) at 36.6%, `afriteva` (229M,
paradoxically not the worst despite being smallest) at 57.9%. Scale
alone does not perfectly order the T5-family group, but architecture
family plus rough scale tier together predict the split almost
exactly across all 12 models -- BART-style or 1B+ T5-family: 74-103%;
base-tier T5-family: 24-58%, with zero overlap between the two bands.

**Flagged as unresolved, not papered over**: why base-tier T5-family
checkpoints specifically respond worse to attention+FFN LoRA than
either BART-style models of any size or larger T5-family models is not
fully explained by anything measured in this benchmark. A plausible
mechanism (smaller pretrained models have less latent capacity for a
low-rank adapter to redirect, so the glossary task -- closer to new
knowledge injection than to steering existing fluency -- needs more
of it) is offered in this section but not verified against a direct
ablation. Worth a follow-up sweep (higher rank, more epochs, or full
FFN fine-tuning restricted to just those three checkpoints) before
treating their LoRA numbers as a ceiling rather than a starting point.

### 11.3 madlad3b / t5-v1_1-xl: unblocked as intended

Both models -- excluded from full fine-tuning by a genuine memory
ceiling (§9.3: untied 256k/32k-vocab embedding tables stored twice,
combined with fp32 optimizer state, leaving ~zero headroom on a 32GB
V100) -- trained cleanly under LoRA, confirming the ceiling was
specifically an optimizer-state problem, not a fundamental capacity
one. Both ran the original `q,v`-only config (§11.2's fix came later,
after these had already finished; not re-run given their already
strong glossary scores, see §12 item 7 for the resulting minor
inconsistency this leaves in the roster).

| Model | BLEU | chrF++ | AfriCOMET | GlossAcc | GlossF1 |
|---|---|---|---|---|---|
| madlad3b | 61.43 | 78.44 | 71.15 | 73.07 | 70.78 |
| t5v11xl | 63.68 | 78.09 | 71.03 | 66.14 | 66.36 |

GPU memory during training: `madlad3b` 19.3GB/32GB, `t5v11xl`
13.2GB/32GB -- comfortable headroom, versus the OOM both hit under
full fine-tuning before even completing a full forward+backward pass.
Adapters saved at 37MB each, versus each checkpoint's ~11.8GB fp32
base weights, confirming `trainer.save_model()` correctly persists
only the adapter for a `PeftModel`. No errors in either log across the
full 5-epoch run.

### 11.4 Remaining roster: complete, all 14 models now have a LoRA result

The other 12 models ran as a chained sweep across both GPUs (the same
pattern as the zero-shot sweep, §9.7 -- pairs launched together, the
chain waits for both before starting the next pair), all under the
fixed `q,v,wi,wo`/`q_proj,v_proj,fc1,fc2` config from §11.2. Order:
`afriteva`+`seamless`, `m2m100`+`mt5_large`, `mt5`+`cheetah`,
`afrimt5`+`toucan`, `nllb`+`afriteva_v2_large`,
`mbart50`+`m2m100_1.2b`. Ran unattended over roughly 71 hours
(Aug 26 20:07 -> Aug 29 19:05) across the 6 pairs, zero crashes, zero
manual intervention beyond the one discard-and-restart in §11.2. Logs
in `logs/lora_sweep/` (per-model `lora_<model>.log`, plus
`lora_chain.log` for pair-level progress); the discarded pre-fix
attempt is preserved as `lora_chain_qv_only.log` rather than deleted.

**Full 14-model LoRA results** (test-set, beam-5, same scoring as
every full-FT/zero-shot table in this report):

| Model | BLEU | chrF++ | GlossAcc | GlossF1 |
|---|---|---|---|---|
| seamless | 65.05 | 79.59 | **78.36** | 76.08 |
| mbart50 | **67.17** | 79.23 | 74.36 | 74.59 |
| m2m100_1.2b | 66.60 | 78.82 | 73.66 | 74.00 |
| nllb | 63.89 | 79.05 | 77.55 | 73.75 |
| madlad3b | 61.43 | 78.44 | 73.07 | 70.78 |
| m2m100 | 65.62 | 78.31 | 69.85 | 70.65 |
| t5v11xl | 63.68 | 78.09 | 66.14 | 66.36 |
| toucan | 64.38 | 78.28 | 62.47 | 64.18 |
| afriteva_v2_large | 65.87 | 78.59 | 59.35 | 62.50 |
| mt5_large | 63.32 | 77.46 | 58.88 | 61.68 |
| cheetah | 63.36 | 77.67 | 57.78 | 60.75 |
| afriteva | 57.50 | 74.96 | 24.17 | 31.26 |
| afrimt5 | 63.56 | 77.61 | 20.51 | 29.22 |
| mt5 | 61.58 | 76.74 | 12.00 | 18.52 |

**This completes RQ4** (full fine-tuning vs. LoRA/PEFT). Two distinct
findings, not one: (1) BLEU/chrF++ are close across the board
regardless of architecture or LoRA-vs-full-FT (57.5-67.2 BLEU, a
narrower band than glossary accuracy's spread) -- LoRA does not
meaningfully cost translation fluency anywhere in the roster; (2) the
glossary-accuracy split documented in §11.2 is real and specific to
that one metric, not visible in BLEU/chrF++ at all -- a model can look
fine on aggregate translation quality while having quietly lost most
of its terminology-glossing behavior, which is exactly why this
benchmark scores glossary-accuracy as an independent metric rather
than trusting BLEU alone (§4.1's original motivation for the metric,
now doubly confirmed by a failure mode BLEU could not see).

`mt5_large` still leads on full-FT GlossF1 (§9.5) and AfriCOMET
(§10.3); under LoRA, `seamless` leads GlossF1 instead (76.08, versus
`mt5_large`'s own LoRA result of 61.68 -- notably, `mt5_large`'s LoRA
score is not even its own best result, full fine-tuning beats it by a
wide margin, consistent with the base-tier-vs-larger split in §11.2
not applying to it specifically, since it is a 1.2B model). LoRA does
not compete for the PidginEdu-LLM designation either way (§9.5), so
none of this reopens that question on metric grounds -- it is settled
as `toucan` regardless, per the qualitative override documented in
§9.5, not because of anything in this section.

---

## 12. Open items before this can be called a finished benchmark

1. ~~Run all 10 planned models~~ -- **done** (§7), then ~~roster
   revised for paradigm diversity and grown to 12~~ -- **also done**
   (§9): `seamless` and `m2m100_1.2b` both landed clean on the first
   attempt, no correction needed. `madlad3b`/`t5-v1_1-xl` hit a genuine
   memory ceiling and are deferred to item 6 below, not abandoned.
   Note `eval_gen_len` is not a reliable cross-architecture signal for
   whether a model is learning to gloss (§7.5) -- usable only within a
   single model's own curve.
2. ~~AfriComet is not yet implemented~~ -- **done** (§10): implemented
   as `africomet_metrics.py`, scored all 12 completed models. `pcm` is
   explicitly in its training language list, unlike almost every other
   multilingual resource used in this project.
3. No HuggingFace dataset release, dataset card, or CC BY 4.0 licensing
   artifacts yet.
4. Terminology-annotation code (i.e., the process that produced
   `Eng-PidginEdu_glossary_augmented.csv` from a plain parallel corpus)
   is not in this repository and is not documented here, since it
   predates this benchmarking work.
5. The NLLB/mBART/M2M-100/MADLAD/SeamlessM4T proxy-language-token
   substitution (§3.12, §9.2) is a genuine confound relative to the
   T5-family models, which see no equivalent identity conflation --
   this should be stated explicitly as a limitation wherever those
   five models' numbers are reported.
6. ~~The zero-shot sweep has not started~~ -- **done** (§9.7): all 14
   models scored (BLEU, chrF++, AfriCOMET, glossary-accuracy), zero
   failures, logs preserved in `logs/zeroshot_sweep/`. Produced the
   fine-tuning-gain table (RQ2/RQ3) -- the inverse relationship between
   zero-shot capability and fine-tuning gain is likely the single most
   paper-relevant finding this benchmark has produced so far.
7. ~~LoRA/PEFT fine-tuning (§9.5, RQ4) has not started~~ -- **done**
   (§11): all 14 models now have a LoRA result. `madlad3b`/
   `t5-v1_1-xl` confirmed unblocked as intended. A real bug
   (attention-only `target_modules` badly degrading the glossary metric
   specifically, §11.2) found and fixed mid-sweep. Final RQ4 finding:
   BART-style models (`seamless`, `mbart50`, `nllb`, `m2m100`,
   `m2m100_1.2b`) retain 95-103% of full-FT glossary accuracy under
   LoRA; the base-tier T5-family models (`mt5`, `afrimt5`, `afriteva`,
   all under 600M) retain only 24-58% -- a clean split with zero overlap
   between the two bands (§11.2/§11.4). This is not yet explained by
   anything measured directly in this benchmark (a capacity hypothesis
   is offered but unverified) -- **flagged as a genuine open question
   for the paper, not resolved by this sweep**. `madlad3b`/
   `t5-v1_1-xl` still carry the pre-fix `q,v`-only config (§11.3) since
   they already scored well before the bug was found -- a documented,
   deliberate inconsistency, not an oversight.
8. ~~"PidginEdu-LLM" is not finalized~~ -- **done, with a disclosed
   late change** (§9.5): the metrics-only process finalized `mt5_large`
   (leads GlossF1/AfriCOMET/chrF++ among 12 full-fine-tuned models,
   confirmed on both test and a corrected validation-set leaderboard).
   LoRA results (§11) never competed for this designation -- full
   fine-tuning represents best-achievable quality, LoRA answers a
   separate cost/adaptability question (RQ4) -- and human evaluation,
   listed as a candidate metric in early planning, was explicitly
   decided against as a deciding criterion at that point. After that
   process completed, the project author reviewed both models' outputs
   directly and overrode it on qualitative grounds (`toucan`'s Pidgin
   judged more natural/fluent) -- **`toucan` is the finalized
   PidginEdu-LLM**, not `mt5_large`. Documented in §9.5 as exactly what
   it is: a single-reviewer judgment call made after and in spite of a
   metrics-only process built to avoid this, not a new evaluation
   protocol. `mt5_large`'s HuggingFace checkpoint was removed at the
   author's request; it remains fully documented here as the
   automated-metric leader.
9. ~~Model selection used test-set scores to pick the winner, a
   multiple-comparison/winner's-curse risk~~ -- **done** (§9.5): an
   external methodology review caught that the original automated
   ranking used test-set performance rather than validation
   performance to pick a leader. Built a full validation-set (dev
   split) leaderboard at matching settings and confirmed `mt5_large`
   wins there too (dev GlossF1/chrF++, same two metrics it led on
   test) -- the automated-metric leader question rests on the
   methodologically correct basis now, not just the original one (the
   PidginEdu-LLM designation itself was subsequently overridden on
   qualitative grounds to `toucan` regardless, item 8 above -- this
   fix concerns the metrics-only ranking, not that later decision).
   This also surfaced and fixed §3.20 (6 checkpoints, including both
   `mt5_large` and `toucan`, silently corrupted on save and unusable on
   reload) -- found only because a checkpoint reload was actually
   attempted for this validation pass, which the original
   sweep never needed to do.
10. ~~Reproducibility packaging is not yet done~~ -- **done**: added
    `requirements.txt` (pinned, verified via `pip check`, with the
    transformers version-drift note from §1), `LICENSE` (MIT, per
    deliverable #5), and `README.md` as a full reproduction guide
    (setup, exact commands for every phase, hardware/time/disk
    expectations, a determinism caveat, and the environment-specific
    gotchas already fixed in this codebase). Found and fixed a real
    portability blocker while doing this: every script hardcoded
    `/home/flora/Eng_Pidgin_Edu` and `/home/flora/env/bin/python` as
    absolute paths, which would have failed immediately for anyone
    cloning to a different location -- `DATA_DIR` in every `.py` file
    now derives from the script's own location
    (`os.path.dirname(os.path.abspath(__file__))`), and every `.sh`
    file resolves `PROJECT_DIR` from `${BASH_SOURCE[0]}` with
    `PYTHON`/`TORCHRUN` overridable via environment variables rather
    than hardcoded. `run_train.sh`'s single-GPU launcher was also still
    using the broken plain-`python` path noted as unfixed in §11.1 --
    fixed to always use `torchrun`. **Verified end-to-end, not just
    asserted**: copied the full non-checkpoint repo contents to a
    throwaway directory (`/tmp/repro_test`, simulating a fresh clone to
    an unrelated path) and ran a real training smoke test from there --
    correct `PROJECT_DIR`/`OUTPUT_DIR` resolution, correct data
    loading, successful `torchrun` launch, real training loss logged,
    zero errors. `.gitignore` (added earlier) still covers
    `output_*/`/`baseline_runs/`/`external/`; `git init` and the first
    commit have not been run yet, pending your go-ahead. No checkpoint
    has been uploaded to HuggingFace yet (deliverable #1) --
    `output_mt5_large/` is the first verified-correct artifact to
    upload once that happens (§3.20 fixed it).

---

## 13. Research questions: answers

The evaluation design (§9.5) was built around six explicit research
questions. This section states each one, points to where its evidence
already lives in this report, and for RQ3/RQ5 -- which had the
supporting data scattered across other sections but no dedicated
synthesis -- works the analysis through directly rather than just
cross-referencing.

### RQ1: How well do existing models perform on Nigerian Pidgin without adaptation?

**Answered in full** -- §9.7, zero-shot sweep, all 14 models, BLEU/
chrF++/AfriCOMET. `seamless` (32.71 BLEU) and both `m2m100` variants
(27-28 BLEU) retain real translation capability with zero Pidgin
exposure, most likely from broader multilingual pretraining coverage;
most T5-family checkpoints (`mt5`, `afrimt5`, `cheetah`) are
essentially non-functional for this pair zero-shot (BLEU < 1).

### RQ2: How much does Eng-PidginEdu improve Nigerian Pidgin translation across model families?

**Answered for 12 of 14 models** -- §9.7's fine-tuning-gain table.
Every model improves substantially (+34.85 to +66.16 BLEU), and the
gains are inversely related to zero-shot starting point -- all 12
converge to a similar final BLEU band (57.7-68.5) regardless of where
they started. This is the report's own assessment of its single most
paper-relevant finding: Eng-PidginEdu's contribution is not
concentrated in "helping already-capable models get better" but in
closing an architecture-independent gap to a common ceiling.

**Gap**: `madlad3b`/`t5-v1_1-xl` have zero-shot scores (§9.7) but no
full-fine-tune score to pair them with (only LoRA, since full
fine-tuning was infeasible for them, §9.3/§11.3), so they are absent
from the gain table. A zero-shot-vs-LoRA gain figure could be added
for these two if 14-model coverage on this specific question matters
for the paper; not done here since it would mix fine-tuning methods
within one table without a clear label, which risks misreading.

### RQ3: Which pretrained model is most adaptable to Nigerian Pidgin?

**Answered here, synthesized from two lenses already in this report
that were not previously combined into a single "adaptability"
statement.**

*By raw fine-tuning gain* (§9.7's table): `cheetah` gains the most
(+66.16 BLEU, from 0.00 zero-shot). But this lens is confounded --
"most adaptable" by this measure mostly just means "worst zero-shot
starting point," since gain and zero-shot score are strongly inversely
related (§9.7). It answers "which model needed the most help,"
not "which model is easiest to adapt."

*By LoRA retention* (§11.2/§11.4): how much of a model's own full-FT
quality survives when adaptation is restricted to a small, efficient
parameter budget is a cleaner adaptability signal -- it measures how
much of the model's capability is reachable through a lightweight
update, independent of where it started. By this measure, `seamless`
(103% retention, exceeding its own full-FT score), `nllb` (97.1%),
`mbart50` (97.9%), and both `m2m100` variants (95.5%) are the most
adaptable models in the roster -- LoRA recovers essentially all of
what full fine-tuning achieves for these. The base-tier T5-family
models (`mt5`, `afrimt5`, `afriteva`) are the least adaptable by this
measure (24-58% retention) -- they need full weight movement, not just
a lightweight nudge, to reach their own best achievable quality.

**Answer**: architecture, not scale, is the dominant factor -- every
BART-style model in the roster (`seamless`, `nllb`, `mbart50`,
`m2m100`, `m2m100_1.2b`) is highly adaptable by the retention measure;
the base-tier T5-family models are consistently the least adaptable,
regardless of the specific checkpoint. If "most adaptable" is read as
"which single model," `seamless` is the strongest answer -- top or
near-top by both lenses (largest zero-shot base a fine-tune builds on
usefully, and the only model whose LoRA result actually exceeds its
own full fine-tune).

### RQ4: Can parameter-efficient adaptation achieve comparable performance to full fine-tuning?

**Answered in full** -- §11, entire section. Short answer: yes for
translation quality broadly (BLEU/chrF++ stay close across the full
roster regardless of method, §11.4), and yes-to-fully-yes for glossary
accuracy specifically on BART-style architectures (95-103% retention),
but no for base-tier T5-family models on the glossary-specific metric
(24-58% retention) -- LoRA is not a uniform substitute for full
fine-tuning, its effectiveness depends on model family.

### RQ5: How does model scale relate to Nigerian Pidgin translation performance?

**Reframed** from the original "600M vs 1.2B vs 3B vs 7B" phrasing --
no 7B-scale model was ever trainable under this pipeline's hardware
(2x32GB V100s): `madlad400-7b-mt`'s 33GB fp32 weights alone exceed a
single GPU's full capacity before any gradient, optimizer, or
adapter state is counted, and that bottleneck is in base-weight
storage, so LoRA does not fix it either (§9.3). The question below
covers the full scale range this benchmark actually ran, 229M-3B, and
is answered directly rather than left as a gap that a missing 7B point
would have filled.

| Model | Params | BLEU | chrF++ | AfriCOMET | GlossF1 |
|---|---|---|---|---|---|
| afriteva | 229M | 57.71 | 74.38 | 63.21 | 49.01 |
| m2m100 | 418M | 66.38 | 78.66 | 71.26 | 73.28 |
| afrimt5 | 580M | 63.41 | 78.19 | 70.82 | 60.32 |
| mt5 | 580M | 62.65 | 77.79 | 70.63 | 56.11 |
| nllb | 600M | 64.61 | 79.60 | 71.50 | 75.88 |
| mbart50 | 680M | 68.18 | 79.96 | 71.80 | 76.74 |
| afriteva_v2_large | 1.0B | 66.32 | 79.57 | 71.62 | 76.54 |
| toucan | 1.2B | 64.74 | 79.34 | 71.80 | 73.96 |
| mt5_large | 1.2B | 66.92 | 80.59 | 71.94 | 78.51 |
| cheetah | 1.2B | 66.16 | 79.62 | 71.84 | 73.16 |
| m2m100_1.2b | 1.24B | 68.50 | 80.14 | 71.76 | 77.20 |
| seamless | 1.37B | 67.56 | 79.74 | 71.92 | 75.52 |
| t5v11xl* | 2.85B | 63.68 | 78.09 | 71.03 | 66.36 |
| madlad3b* | 3B | 61.43 | 78.44 | 71.15 | 70.78 |

*LoRA result, not full fine-tune -- these two never got a full-FT run
(§9.3), so they are not a strictly apples-to-apples scale comparison
with the rest of this table; included for completeness with the
caveat stated, not silently blended in.

**Finding: a sharp threshold effect from 229M to ~400-600M, then a
plateau with no consistent further gain from scale alone.** `afriteva`
(229M) is the clear outlier, well below every other model on every
metric. From 418M upward, all four metrics cluster tightly regardless
of further scale increases (BLEU 61.4-68.5, chrF++ 77.5-80.6,
AfriCOMET 70.6-72.0, GlossF1 56.1-78.5) -- `mbart50` at 680M beats
several 1.2B+ models on BLEU and chrF++, and the two largest models
tested (2.85B/3B, `t5v11xl`/`madlad3b`, LoRA-only) do not outperform
the 600M-1.4B tier on any metric. Pearson correlation between
parameter count and each metric across the 12 full-FT models confirms
this quantitatively: **0.62-0.72 including `afriteva`, dropping to
0.49-0.75 once it's excluded** -- most of the apparent scale
relationship is really one small-model penalty, not a continuous
scaling trend among the mid-to-large checkpoints. **Architecture and
fine-tuning approach explain far more of the variance in this dataset
than raw parameter count does once a model clears roughly 400-600M
parameters.**

### RQ6: Can PidginEdu-LLM establish a competitive specialized model for Nigerian Pidgin?

**Answered for the internal comparison, with a caveat worth stating
plainly** -- §9.5: the finalized PidginEdu-LLM is `toucan`, selected by
the project author's qualitative judgment of translation quality, not
by the automated metrics this benchmark otherwise uses throughout.
`mt5_large` is the model that actually leads 3 of 4 automated metrics
(GlossF1 78.51, AfriCOMET 71.94, chrF++ 80.59) against the other 13
models in the roster. So this benchmark's own automated evidence
answers "which model is most competitive against the other 14" with
`mt5_large`; the "PidginEdu-LLM" label answers a related but distinct
question -- which model the author chose to release as the flagship --
and those two answers now point at different checkpoints, disclosed
here rather than blended into one.

**Not answered**: comparison against external, previously-published
Nigerian Pidgin MT systems (i.e., "competitive" in the sense of the
broader field, not just this benchmark's own roster). The one attempt
at an external baseline in this project -- Pidgin-UNMT (§9.1) -- was
investigated and dropped: its linked pretrained checkpoint is no
longer retrievable (empty Google Drive folder, confirmed directly),
and no other pretrained Nigerian Pidgin MT system was identified as
retrievable and runnable. If RQ6 is meant to include external
competitiveness, that remains open and would need either a different
external system to be found runnable, or a citation-based comparison
against published numbers from prior Nigerian Pidgin MT work instead
of a re-run comparison.