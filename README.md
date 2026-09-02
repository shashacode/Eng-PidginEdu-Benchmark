# Eng-PidginEdu: A Glossary-Augmented English→Nigerian Pidgin MT Benchmark

Fine-tuning, zero-shot, and LoRA/PEFT benchmark of 14 multilingual MT
models on English→Nigerian Pidgin translation with inline terminology
glossing. Full methodology, every bug found and fixed, every
hyperparameter, and every result is documented in
[`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) -- that file is the
canonical record of this project; this README is a reproduction guide.

**Research questions answered** (BENCHMARK_REPORT.md §13): zero-shot
capability, the value of fine-tuning, which model adapts most
efficiently under LoRA, full fine-tuning vs. parameter-efficient
adaptation, model scale vs. performance, and how the resulting
flagship model compares against the other 13. `mt5_large` is the
automated-metric leader (leads GlossF1/AfriCOMET/chrF++); the finalized
PidginEdu-LLM is `toucan`, selected by the project author's qualitative
judgment of translation quality over the automated-metric leader --
see §9.5 for the full, disclosed reasoning behind that override.

## 1. What you need before starting

- **2 GPUs with ≥32GB VRAM each** for the full-fine-tuning phase as
  originally run (a single 32GB+ GPU works for everything except the
  1B+ tier, which needs Adafactor + gradient checkpointing either way
  -- see §6 of the report). LoRA and zero-shot both fit comfortably on
  a single GPU per model.
- **~250GB free disk** if you train every model and keep every
  checkpoint (each full-fine-tune checkpoint is 6–17GB; see §7.4 for
  per-model training times). You do not need this much to reproduce
  any *individual* result -- see §4 below for scoped reproduction.
- **A HuggingFace account + token** (`huggingface-cli login`, or set
  `HF_TOKEN`). `UBC-NLP/cheetah-1.2B` is manually gated (report §3.15)
  -- request access on its model page before attempting that model
  specifically; every other checkpoint used here is open.
- This was developed and verified on Tesla V100s (compute capability
  7.0, Volta -- **no bf16 support**, which is why everything runs in
  fp32; report §1/§3.1). On newer hardware (A100/H100) you can likely
  use bf16 for a real speedup, but that is untested here and may shift
  numbers slightly -- fp32 is what every result in this report was
  produced with.

## 2. Setup

```bash
git clone <this-repo-url>
cd Eng-PidginEdu

python -m venv env
source env/bin/activate

# torch needs the CUDA wheel index, not plain PyPI
pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

See [`requirements.txt`](requirements.txt) for the full pinned list
and a note on a transformers version drift discovered while packaging
this (5.15.0 → 4.57.6 partway through the project, verified to not
affect any reported result -- BENCHMARK_REPORT.md §1).

**One environment-specific caveat, not universal**: torch 2.13
dispatches some generation-time ops through Triton, which JIT-compiles
against the CPython headers (`Python.h`). If your system doesn't have
`python3-dev` (or equivalent) installed, training will run fine and
then fail at the first evaluation. `run_train.sh` handles this with a
best-effort fallback and a clear warning if headers aren't found
anywhere; the simplest real fix on most systems is just:

```bash
sudo apt install python3-dev   # or your distro's equivalent
```

## 3. The dataset

`Eng-PidginEdu_glossary_augmented.csv` (26,232 English–Pidgin pairs,
61.4% carrying inline terminology glosses like `acceptable (dem
worthy of acceptance...)`) is included in this repo, already split
80/10/10 into `train.json` (20,986) / `dev.json` (2,623) /
`test.json` (2,623) -- report §2 has the full breakdown by subject and
gloss statistics. The canonical hosted release (CC BY 4.0) is at
[huggingface.co/datasets/coderGit/Eng_PidginEdu](https://huggingface.co/datasets/coderGit/Eng_PidginEdu).

If you ever need to rebuild the split files from the CSV (they're
already here, so you shouldn't need to):

```bash
python prepare_data.py
```

To rebuild the glossary-augmented CSV itself from the raw parallel
corpus (`Eng-PidginEdu_Dataset.csv`) and the standalone terminology
table (`academic_glossary.csv`), both also included here:

```bash
python glossary_augment.py
```

Verified to reproduce `Eng-PidginEdu_glossary_augmented.csv` exactly,
byte-for-byte across all 26,232 rows -- report §12 item 4 has the full
verification, including two non-obvious behaviors that had to be
reverse-engineered (a first-occurrence-only rule for duplicate glossary
entries, and a pandas NA-string quirk affecting one row) rather than
assumed from reading the algorithm alone.

## 4. Reproducing results

Every command below writes to `output_<model>/` (full fine-tune),
`output_lora_<model>/` (LoRA), or `output_zeroshot_<model>/`
(zero-shot) -- each a self-contained result: `test_predictions.csv`,
`glossary_report.json` (BLEU/chrF++/TER/glossary-accuracy), and
`metrics.json`. **These `output_*` directories are the actual results
this report cites, and the only ones published in this repository.**
Report §3/§5/§7 also describe several superseded, pre-fix runs (a
wrong learning rate, a wrong prefix, an invalidated language token) as
before/after evidence for those incidents -- those prediction files
exist only on the author's own machine, deliberately excluded from
version control, so you will not find them by cloning this repo; the
incidents themselves are still fully documented in report §3/§5/§7/§8
even though the raw files aren't included.

**Model keys** (pass to `--model`): `afriteva`, `m2m100`, `mt5`,
`afrimt5`, `nllb`, `mbart50`, `afriteva_v2_large`, `mt5_large`,
`cheetah`, `toucan`, `seamless`, `m2m100_1.2b`, `madlad3b`, `t5v11xl`.
Full per-model hyperparameters (learning rate, batch size, optimizer,
LoRA config) are in report §6 and §11.1 -- every run below uses each
model's own tuned defaults, so no extra flags are needed to match this
report's numbers.

### 4a. Full fine-tuning (12 of the 14 models)

```bash
# Single model, all visible GPUs auto-detected:
./run_train.sh mt5_large

# Force a GPU count, or pin specific GPUs:
NUM_GPUS=1 ./run_train.sh afriteva
CUDA_VISIBLE_DEVICES=0 ./run_train.sh afriteva

# The whole 12-model sweep, cheapest model first, skipping anything
# already finished (safe to re-run/resume):
./run_benchmark.sh
```

`madlad3b` and `t5v11xl` are **not** in this sweep -- they hit a
genuine GPU memory ceiling under full fine-tuning (report §9.3) and
only support LoRA. Attempting `./run_train.sh madlad3b` without
`--lora` will reproduce that same OOM, not a new bug.

### 4b. LoRA/PEFT fine-tuning (all 14 models)

```bash
./run_train.sh madlad3b --lora     # the two models this unblocks
./run_train.sh t5v11xl --lora

./run_train.sh mt5_large --lora    # any of the other 12 too
```

LoRA hyperparameters (`r=16`, `alpha=32`, `dropout=0.05`, forced
`adamw_torch` + `1e-3` LR, and per-architecture `target_modules`) are
documented in full -- including a real bug found and fixed mid-project
(attention-only target modules badly degrading the glossary metric
specifically) -- in report §11.1–§11.2. Override any of them with
`--lora-r`, `--lora-alpha`, `--lora-dropout`, `--lora-target-modules`.

### 4c. Zero-shot evaluation (no fine-tuning, all 14 models)

```bash
python evaluate_model.py --model-key mt5_large --zero-shot --num-beams 1
```

Writes to `output_zeroshot_mt5_large/`. Report §9.7 used greedy
decoding (`--num-beams 1`) uniformly across all 14 models as a
deliberate methodological choice -- change it if you want, but it will
no longer match the report's zero-shot table directly.

### 4d. Re-scoring an already-fine-tuned checkpoint

Useful for scoring against a different split, or verifying a
checkpoint reproduces its published numbers:

```bash
python evaluate_model.py --model-key mt5_large \
    --model-dir output_mt5_large --split test --num-beams 5
```

**If you score anything other than `--split test`, always pass
`--out-dir` explicitly** (e.g. `--out-dir output_mt5_large_dev`) -- the
default output location is the same directory as `--model-dir`, and
without an override you will silently overwrite the original test-set
results with whatever split you just scored.

### 4e. AfriCOMET scoring

Requires a model's predictions to already exist (run 4a/4b/4c first):

```bash
python africomet_metrics.py --model-dir output_mt5_large

# Every finished run in the project directory at once:
python africomet_metrics.py --all
```

### 4f. Aggregating everything into one leaderboard

```bash
python aggregate_results.py
```

Writes `benchmark_results.csv` / `benchmark_results.md`, sorted by
GlossF1 by default (`--sort <column>` to change it). This scans
*every* `output_*/glossary_report.json` it can find, so full-FT,
LoRA, zero-shot, and dev-split re-score results all land in one table
together for the same model -- the `Condition` column in
`benchmark_results.md` (derived from the directory naming convention,
since not every run type writes it to its own report) says which is
which; filter on it, or on the raw `output_dir` column in the CSV, if
you want just one condition.

## 5. What to expect

Full fine-tuning: roughly 1.5–8 hours per model on 2×V100-32GB
depending on size and optimizer tier (report §7.4 has the exact
per-model wall-clock table). LoRA runs on the same hardware took
roughly 9–13 hours per model in this project's own runs (single GPU
each, report §11.4) -- LoRA does not reduce wall-clock time here, only
trainable-parameter count and therefore GPU memory; it does not speed
up the forward/backward pass through the frozen base model.

**Determinism**: `SEED = 42` is fixed for Python/NumPy/PyTorch
throughout (`train.py`), so results should be close to exactly
reproducible run-to-run *on the same GPU architecture and library
versions*. Exact bit-identical output across different GPU models,
driver versions, or library versions is not guaranteed -- this is a
property of floating-point non-associativity in GPU kernels generally,
not something specific to this codebase, and is the standard
expectation for ML reproducibility (matching within noise, not to the
bit).

## 6. Known environment-specific gotchas (all documented, all fixed here)

These were all found and fixed during this project and are already
handled in this codebase -- listed here so a re-run doesn't look like
it's failing when it's actually working as designed:

- Single-GPU runs go through `torchrun --nproc_per_node=1`, not plain
  `python` -- a known `accelerate` bug otherwise crashes with `Default
  process group has not been initialized` (report §11.1).
- `UBC-NLP/toucan-1.2B` and `UBC-NLP/cheetah-1.2B` need a custom
  tokenizer loading path (`custom_tokenizer` in `MODEL_REGISTRY`) --
  their declared `tokenizer_class` doesn't match the actual tokenizer
  file (report §3.16).
- `forced_bos_token_id` must be set on `model.generation_config`, not
  `model.config` (report §3.10).
- A checkpoint-corruption bug (report §3.20) affecting 6 of 12
  full-fine-tuned models' saved `tie_word_embeddings` config was found
  and fixed in this repo's checkpoints and guarded against for any
  future run of `train.py` -- no action needed on your part, just
  documented here so the history is clear.

## 7. Repository layout

```
train.py                 Full fine-tuning + LoRA (--lora flag)
evaluate_model.py         Zero-shot and re-scoring of saved checkpoints
glossary_metrics.py       The glossary-accuracy metric (report §4)
glossary_augment.py       Terminology-annotation preprocessing (report §12 item 4)
africomet_metrics.py      AfriCOMET scoring (report §10)
aggregate_results.py      Leaderboard builder
prepare_data.py           Rebuilds train/dev/test.json from the CSV
run_train.sh               Single-model launcher (handles GPU detection)
run_benchmark.sh          Full 12-model full-fine-tune sweep
BENCHMARK_REPORT.md       Full methodology, every result, every bug found
PAPER_DRAFT.md            Research paper draft (not submission-ready -- see its
                          own status note)
```

## License

MIT -- see [`LICENSE`](LICENSE).
