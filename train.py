# ============================================================
# UNIVERSAL NIGERIAN PIDGIN MT FINETUNING PIPELINE
# ============================================================
# Supports:
# - MT5
# - ByT5
# - AfriMT5
# - AfriByT5
# - NLLB
# - AfriTeVa
# - Toucan
#
# Outputs:
# - BLEU
# - chrF++
# - TER
# - predictions.csv
# - metrics.json
# - checkpoints
# - fine-tuned model
#
# Reproducible
# Standalone
# No run_translation.py required
#
# Single GPU:
#     python train.py --model afrimt5
#
# Multi GPU (DistributedDataParallel, one process per GPU):
#     torchrun --standalone --nproc_per_node=2 train.py --model afrimt5
#
# Or just use ./run_train.sh, which detects the GPU count for you.
# ============================================================

# ============================================================
# INSTALL
# ============================================================

# !pip install transformers datasets evaluate sacrebleu accelerate sentencepiece pandas -q

# ============================================================
# IMPORTS
# ============================================================

import os
import gc
import json
import torch
import random
import argparse
import evaluate
import numpy as np
import pandas as pd

import torch.distributed as dist

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast,
    SeamlessM4Tv2ForTextToText,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from huggingface_hub import hf_hub_download

from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# DISTRIBUTED CONTEXT
# ============================================================
# torchrun sets RANK / LOCAL_RANK / WORLD_SIZE. When absent we are
# running as a plain single-process job and every rank check is a no-op.

RANK = int(os.environ.get("RANK", 0))

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

IS_MAIN = RANK == 0


def log(message):
    """Print once per job rather than once per GPU."""
    if IS_MAIN:
        print(message, flush=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# MODEL CONFIGS
# ============================================================
# batch_size is PER GPU. effective_batch is the global batch we want
# regardless of how many GPUs are in play; gradient accumulation is
# derived from it below so results stay comparable across GPU counts.

# ------------------------------------------------------------
# A NOTE ON LANGUAGE CODES
# ------------------------------------------------------------
# None of the multilingual MT models here support Nigerian Pidgin as a
# target language. Verified against the shipped tokenizers:
#
#   NLLB-200   202 language codes, pcm_Latn ABSENT (has hau/ibo/yor)
#   mBART-50    52 language codes, pcm      ABSENT
#   M2M-100    100 language codes, pcm      ABSENT
#
# Fine-tuning therefore has to repurpose an existing language token. The
# default is Tok Pisin (tpi_Latn) for NLLB: it is the only English-lexified
# creole in the inventory, and unlike eng_Latn it gives the decoder a target
# token distinct from the source language, which avoids biasing the model
# toward copying the English input verbatim. mBART-50 and M2M-100 have no
# creole at all, so they fall back to their English token.
#
# This is a documented modelling choice, not a detail -- state it in the
# paper. Override with --tgt-lang-code.
# ------------------------------------------------------------
#
# batch_size is PER GPU. effective_batch is the global batch we want
# regardless of how many GPUs are in play; gradient accumulation is
# derived from it below so results stay comparable across GPU counts.
#
# optim: the 1B+ models do not fit on a 32GB V100 with fp32 AdamW
# (16 bytes/param). Adafactor's factored second moments bring that down
# to roughly 8 bytes/param, which is what makes that tier trainable here.

MODEL_CONFIGS = {

    # ---- comfortably fits: fp32 AdamW on a 32GB V100 ----

    "afriteva": {
        "model_name": "castorini/afriteva_base",
        "params": "229M",
        "source_prefix": "translate English to Pidgin: ",
        "batch_size": 8,
        "effective_batch": 16,
        "learning_rate": 3e-5,
        "ddp_find_unused_parameters": False,   # confirmed safe: trained clean
    },
    "m2m100": {
        "model_name": "facebook/m2m100_418M",
        "params": "418M",
        "source_prefix": "",
        "lang_style": "m2m100",
        "src_lang_code": "en",
        "tgt_lang_code": "en",      # no creole in the 100-language inventory
        "batch_size": 8,
        "effective_batch": 16,
        "learning_rate": 3e-5,
        # M2M100Attention names its projections q_proj/k_proj/v_proj/
        # out_proj (confirmed via named_modules()), not T5Attention's
        # bare q/k/v/o -- the CLI default only matches the T5 family.
        "lora_target_modules": "q_proj,v_proj,fc1,fc2",
    },
    "mt5": {
        "model_name": "google/mt5-base",
        "params": "580M",
        "source_prefix": "translate English to Pidgin: ",
        "batch_size": 4,
        "effective_batch": 16,
        # Raised from 5e-5. At the original rate this underfit badly
        # (test_loss 1.16 vs afriteva's 0.60) and never learned to emit
        # terminology glosses at all -- 29 produced against 2726 expected.
        # T5-family fine-tuning conventionally wants far more than 5e-5.
        "learning_rate": 1e-4,
        "ddp_find_unused_parameters": False,   # confirmed safe: trained clean
    },
    "afrimt5": {
        "model_name": "masakhane/afri-mt5-base",
        "params": "580M",
        "source_prefix": "translate English to Pidgin: ",
        "batch_size": 4,
        "effective_batch": 16,
        # Raised from 3e-5, same reasoning: best BLEU of the three but
        # only 42 glosses produced against 2726 expected.
        "learning_rate": 1e-4,
        "ddp_find_unused_parameters": False,   # confirmed safe: trained clean
    },
    # nllb's first run at 1e-5 ran the FULL 6560 steps and converged
    # (eval_loss plateaued at 0.985-0.999 across its last 5 evaluations,
    # not still descending) yet still produced almost no glosses
    # (GlossAcc 0.26, 78/2726 -- see BENCHMARK_REPORT.md section 7.2).
    # This differs from the mt5/afrimt5 case: that was cut short before
    # convergence by low early-stopping patience; this converged to a
    # genuine local optimum that ignores the glossing objective.
    # Raising the rate is the next lever to test, but unlike the T5-family
    # fix this is NOT yet a confirmed diagnosis -- it may not transfer,
    # since NLLB is a different architecture converging normally rather
    # than visibly underfitting. The original 1e-5 run's full results are
    # preserved in results_baseline_lr1e5/nllb/ regardless of outcome.
    "nllb": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "params": "600M",
        "source_prefix": "",
        "lang_style": "nllb",
        "src_lang_code": "eng_Latn",
        "tgt_lang_code": "tpi_Latn",   # English-lexified creole proxy
        "batch_size": 4,
        "effective_batch": 16,
        "learning_rate": 1e-4,
        # Same BART-style attention naming as m2m100 above -- confirmed
        # directly for this checkpoint too, not just assumed by family.
        "lora_target_modules": "q_proj,v_proj,fc1,fc2",
    },
    # Never yet run. Preemptively raised for the same reason as nllb
    # above, to avoid discovering the identical failure a second time on
    # a full first run -- also unconfirmed.
    "mbart50": {
        "model_name": "facebook/mbart-large-50-many-to-many-mmt",
        "params": "680M",
        "source_prefix": "",
        "lang_style": "mbart",
        "src_lang_code": "en_XX",
        "tgt_lang_code": "en_XX",   # no creole in the 50-language inventory
        "batch_size": 4,
        "effective_batch": 16,
        "learning_rate": 1e-4,
        "lora_target_modules": "q_proj,v_proj,fc1,fc2",
    },

    # ---- tight: needs Adafactor + gradient checkpointing ----

    # Adafactor at 1e-5 was even more conservative relative to T5's own
    # fine-tuning convention (~1e-3 with Adafactor) than the 3e-5/5e-5
    # AdamW models were before that underfit and failed to learn glossing
    # entirely (see BENCHMARK_REPORT.md section 5). Raised to 1e-3 here
    # pre-emptively rather than re-discovering the same failure at 1.2B
    # scale. Unlike the mt5/afrimt5 fix this has not yet been empirically
    # verified on this data -- watch the first few evaluations' gen_len
    # and gloss counts, not just loss, before trusting a run at face value.
    "afriteva_v2_large": {
        "model_name": "castorini/afriteva_v2_large",
        "params": "1B",
        "source_prefix": "translate English to Pidgin: ",
        "batch_size": 2,
        "effective_batch": 16,
        "learning_rate": 1e-3,
        "optim": "adafactor",
    },
    "mt5_large": {
        "model_name": "google/mt5-large",
        "params": "1.2B",
        "source_prefix": "translate English to Pidgin: ",
        "batch_size": 2,
        "effective_batch": 16,
        "learning_rate": 1e-3,
        "optim": "adafactor",
    },
    "cheetah": {
        "model_name": "UBC-NLP/cheetah-1.2B",
        "params": "1.2B",
        "source_prefix": "translate English to Pidgin: ",
        "batch_size": 2,
        "effective_batch": 16,
        # Pulled back from 1e-3, same as toucan's fix (see toucan's
        # comment below). Confirmed here as genuinely a learning-rate
        # problem, not the prefix: cheetah's prefix was already this
        # correct string from the start, yet it showed the identical
        # failure signature at 1e-3 (eval BLEU ~0, eval_loss stuck in
        # the 4-6 range, wildly erratic gen_len) that toucan showed
        # before ITS prefix was ALSO fixed -- isolating LR as the
        # actual cause for both UBC-NLP checkpoints, not the tokenizer.
        "learning_rate": 1e-4,
        "optim": "adafactor",
    },
    "toucan": {
        "model_name": "UBC-NLP/toucan-1.2B",
        "params": "1.2B",
        # "<2pcm> " (the original choice here) is NOT a registered token
        # in this checkpoint's vocabulary -- checked directly against
        # tokenizer.json's 103 added_tokens (pad/eos/unk + 100 T5 sentinel
        # tokens, no language tags at all). It silently fragmented into
        # 5 meaningless BPE pieces ("<", "2", "p", "cm", ">") on every
        # input. Switched to the plain-text instruction prefix that
        # every other T5-family model in this sweep already uses
        # successfully. See BENCHMARK_REPORT.md for the full incident.
        "source_prefix": "translate English to Pidgin: ",
        "batch_size": 2,
        "effective_batch": 16,
        # Pulled back from 1e-3: that rate, identical to
        # afriteva_v2_large/mt5_large which both converged cleanly,
        # left THIS checkpoint stuck at eval BLEU < 1 through 65% of
        # training (loss plateaued near 3.0, never approaching the
        # <0.5 every other model reached) while generating fluent but
        # source-unrelated text -- a sign of losing pretrained ability
        # under too-aggressive fine-tuning, not "hasn't learned yet".
        # 1e-4 is untested for this checkpoint; watch the first few
        # evaluations closely rather than assuming this transfers.
        "learning_rate": 1e-4,
        "optim": "adafactor",
    },

    # ---- paradigm-diversity additions: different labs, different
    # pretraining objectives, not all previously exposed to Pidgin ----

    # madlad3b and t5v11xl both hit a genuine full-fine-tune memory
    # ceiling (§9.3/§9.5, and the excluded-block comment further down
    # keeps the full incident detail) -- untied embeddings plus fp32
    # AdamW/Adafactor state left essentially zero headroom on a 32GB
    # GPU. Re-added here for LoRA only: with ~99% of parameters frozen,
    # there's no optimizer state for the bulk of the model and gradient
    # memory drops by the same fraction, which should clear the ceiling
    # that blocked full fine-tuning. batch_size/effective_batch/
    # learning_rate below are full-FT-shaped placeholders (never used
    # without --lora, which overrides OPTIM and LEARNING_RATE anyway --
    # see the LoRA default wiring above); NOT confirmed to fit even
    # under LoRA until actually run.

    "madlad3b": {
        "model_name": "google/madlad400-3b-mt",
        "params": "3B (untied input/output embeddings -- stored twice)",
        # "<2kri>" (Krio): verified to tokenize as a single real vocab
        # entry (id 300), not fragmented -- MADLAD has no pcm tag among
        # its 493 languages, and Krio is the closest linguistic relative
        # to Nigerian Pidgin of any tag available (§9.3).
        "source_prefix": "<2kri> ",
        "batch_size": 4,
        "effective_batch": 16,
        "learning_rate": 1e-4,
        "optim": "adafactor",
    },
    "t5v11xl": {
        "model_name": "google/t5-v1_1-xl",
        "params": "2.85B (untied input/output embeddings -- stored twice)",
        # Monolingual English-only pretraining (C4 span-corruption, no
        # translation task or non-English exposure) -- zero-shot/LoRA
        # results here measure how much fine-tuning alone can teach both
        # a new language and a new task (§9.5).
        "source_prefix": "translate English to Pidgin: ",
        "batch_size": 4,
        "effective_batch": 16,
        "learning_rate": 1e-4,
        "optim": "adafactor",
    },

    "seamless": {
        "model_name": "facebook/seamless-m4t-v2-large",
        "params": "1.37B (text-only; discards the multimodal checkpoint's speech/vocoder weights)",
        "source_prefix": "",
        "lang_style": "seamless",
        # No pcm in SeamlessM4T's ~100-language list (checked directly:
        # 102 __xxx__ tags, includes Igbo/Yoruba but no creole at all,
        # unlike NLLB/madlad which at least had a West African creole
        # option). No usable proxy exists here, so this falls back to
        # English -- same fallback rule as mbart50/m2m100 (§ their
        # configs), stated the same way: a genuine confound, not a detail.
        "src_lang_code": "__eng__",
        "tgt_lang_code": "__eng__",
        # 1.37B loaded (not the full 9.2GB checkpoint -- see the model-
        # loading comment near SeamlessM4Tv2ForTextToText). Measured
        # directly: fp32 forward+backward at batch_size=1 uses 11.0GB,
        # comfortably under budget, so batch_size=2 is used here rather
        # than the 1B+ tier's usual batch_size=1 floor -- untested at
        # this exact setting (measurement was at batch 1); watch the
        # first evaluation's memory behavior before trusting a longer run.
        "batch_size": 2,
        "effective_batch": 16,
        "learning_rate": 1e-4,
        "optim": "adafactor",
        # SeamlessM4Tv2Attention names its projections q_proj/v_proj too
        # (confirmed via named_modules() on this exact checkpoint) --
        # same BART-style convention as m2m100/mbart50/nllb, not T5's.
        "lora_target_modules": "q_proj,v_proj,fc1,fc2",
    },

    "m2m100_1.2b": {
        "model_name": "facebook/m2m100_1.2B",
        "params": "1.24B",
        # Same lang_style/mechanism as the already-completed m2m100
        # (418M) config: M2M100ForConditionalGeneration, "en" target
        # since M2M-100's 100-language inventory has no creole (checked
        # for the 418M variant, §3.12; same tokenizer/vocab family here).
        # Chosen over the two 12B checkpoints (facebook/m2m100-12B-*),
        # which are d_model=4096 -- roughly 1.7x the size of the
        # madlad400-7b-mt that already OOM'd outright (§9.3) -- clearly
        # infeasible for full fine-tuning under this pipeline.
        "source_prefix": "",
        "lang_style": "m2m100",
        "src_lang_code": "en",
        "tgt_lang_code": "en",
        # ~5.0GB fp32 weights -- same scale as seamless's 5.5GB, which
        # trained cleanly at batch_size=2 (confirmed, §9.4), so used here
        # too rather than the 1B+ tier's more conservative batch=1 floor.
        "batch_size": 2,
        "effective_batch": 16,
        "learning_rate": 1e-4,
        "optim": "adafactor",
        # ddp_find_unused_parameters left at the pipeline default (True)
        # deliberately -- confirmed necessary for the M2M-100 family
        # specifically (§3.9); do not set this to False for this model.
        "lora_target_modules": "q_proj,v_proj,fc1,fc2",
    },
}

# Excluded from FULL fine-tuning, deliberately (madlad3b/t5v11xl have an
# active MODEL_CONFIGS entry above, but only for --lora runs -- selecting
# either without --lora reproduces the exact OOM detailed below):
#   Pidgin-UNMT 220M   -- no HF checkpoint; separate XLM/UnsupervisedMT
#                         codebase (not transformers-compatible), and its
#                         own linked pretrained checkpoint is no longer
#                         retrievable (empty Google Drive folder, checked
#                         directly) -- see BENCHMARK_REPORT.md for the
#                         full investigation.
#   mT5-XL 3.7B        -- 59GB fp32 AdamW, needs ZeRO-3/FSDP offload;
#                         superseded by the paradigm-diversity additions
#                         above rather than pursued directly.
#   Cheetah-3.7B 3.7B  -- same, superseded likewise.
#   TranslateGemma-4B  -- decoder-only, gated, bf16-native (V100 has none).
#   madlad400-7b-mt    -- measured directly: 33GB fp32 weights alone
#                         exceeds a single 32GB GPU's budget before
#                         gradients, optimizer state, or activations are
#                         counted (~60GB minimum total under this
#                         pipeline's DDP-replicate-per-GPU design). Same
#                         class of blocker as mT5-XL/Cheetah-3.7B above;
#                         needs FSDP/DeepSpeed ZeRO sharding across both
#                         GPUs to attempt, which this pipeline does not
#                         yet implement.
#   madlad400-3b-mt    -- attempted directly, twice (once at max_length
#                         200, once at 128 to rule out sequence length as
#                         the cause -- identical OOM both times, at the
#                         identical 31.23GB allocated). This checkpoint's
#                         input/output embeddings are genuinely untied
#                         (confirmed via the loader's own warning: "both
#                         are present in the checkpoints with different
#                         values"), so its 256k-vocab embedding table is
#                         stored twice rather than shared -- on top of
#                         plain fp32 weights (11.8GB) + fp32 gradients
#                         (11.8GB) + Adafactor state, this leaves
#                         essentially zero headroom on a 32GB GPU before
#                         any activation memory at all. Sequence length
#                         and batch size were not the bottleneck; this is
#                         a fixed-footprint ceiling. Revisit under LoRA.
#                         Proxy target token if revisited: "<2kri>"
#                         (Krio) -- verified to tokenize as a single real
#                         vocab entry (id 300), not fragmented, since
#                         MADLAD has no pcm tag among its 493 languages
#                         and Krio is the closest linguistic relative to
#                         Nigerian Pidgin of any tag available.
#   t5-v1_1-xl 2.85B   -- not run to failure directly (dropped once
#                         madlad3b's identical untied-embedding warning
#                         appeared for this checkpoint too, at a near-
#                         identical size, making the same OOM near-
#                         certain) -- also monolingual English-only
#                         (C4 span-corruption, no translation-task or
#                         non-English exposure during pretraining), so
#                         zero-shot/PEFT results here would measure how
#                         much fine-tuning alone can teach both a new
#                         language and a new task. Revisit under LoRA.

# ============================================================
# ARGUMENTS
# ============================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(
    description="Finetune a seq2seq model for English -> Nigerian Pidgin."
)

parser.add_argument("--model", default="afrimt5", choices=sorted(MODEL_CONFIGS))

parser.add_argument("--train-file", default=os.path.join(DATA_DIR, "train.json"))
parser.add_argument("--val-file",   default=os.path.join(DATA_DIR, "dev.json"))
parser.add_argument("--test-file",  default=os.path.join(DATA_DIR, "test.json"))

parser.add_argument("--output-dir", default=None,
                    help="Defaults to <data dir>/output_<model>.")

parser.add_argument("--epochs", type=int, default=5)

parser.add_argument("--batch-size", type=int, default=None,
                    help="Per-GPU batch size; overrides the model config.")

parser.add_argument("--effective-batch", type=int, default=None,
                    help="Global batch size; gradient accumulation is derived from it.")

parser.add_argument("--learning-rate", type=float, default=None)

parser.add_argument("--num-beams", type=int, default=5,
                    help="Beams for the final test predictions.")

parser.add_argument("--eval-num-beams", type=int, default=1,
                    help="Beams for the per-epoch validation passes. Greedy "
                         "is far cheaper and ranks checkpoints just as well; "
                         "the reported test scores still use --num-beams.")

parser.add_argument("--eval-batch-size", type=int, default=None,
                    help="Defaults to twice the train batch: generation "
                         "carries no optimizer state.")

parser.add_argument("--target-column", default="pcm_augmented",
                    choices=["pcm", "pcm_augmented", "pcm_clean"],
                    help="Training target. pcm_augmented carries the inline "
                         "terminology glosses; pcm_clean is the plain "
                         "translation, for ablations against the glossary.")

parser.add_argument("--max-source-length", type=int, default=200)
parser.add_argument("--max-target-length", type=int, default=200)

parser.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="Faster, but needs noticeably more GPU memory.")

parser.add_argument("--group-by-length", action="store_true",
                    help="Batch similar-length examples to cut padding waste.")

parser.add_argument("--num-workers", type=int, default=4,
                    help="Dataloader workers per process.")

parser.add_argument("--src-lang-code", default=None,
                    help="Override the source language token (NLLB/mBART/M2M-100).")

parser.add_argument("--tgt-lang-code", default=None,
                    help="Override the target language token. Nigerian Pidgin is "
                         "absent from all three inventories, so this selects the "
                         "proxy token that gets repurposed.")

parser.add_argument("--optim", default=None,
                    help="Override the optimizer (e.g. adafactor for the 1B+ tier).")

parser.add_argument("--lora", action="store_true",
                    help="Train a LoRA adapter instead of full fine-tuning. "
                         "Freezes the base model and only updates low-rank "
                         "update matrices injected into the attention "
                         "projections -- ~1%% of parameters, which drops "
                         "optimizer-state memory enough to unblock models "
                         "that hit a full-fine-tune OOM ceiling (madlad3b, "
                         "t5v11xl). Also raises the default learning rate, "
                         "forces adamw_torch (Adafactor's memory benefit is "
                         "moot with this few trainable params), and writes "
                         "to output_lora_<model> instead of output_<model> "
                         "so results never collide with a full fine-tune.")

parser.add_argument("--lora-r", type=int, default=16,
                    help="LoRA rank: width of the low-rank update matrices.")

parser.add_argument("--lora-alpha", type=int, default=32,
                    help="LoRA scaling factor, applied as alpha/r.")

parser.add_argument("--lora-dropout", type=float, default=0.05)

parser.add_argument("--lora-target-modules", default=None,
                    help="Comma-separated module names to attach LoRA "
                         "adapters to. Defaults to each model's own "
                         "lora_target_modules config entry -- 'q,v,wi,wo' "
                         "for the T5-family checkpoints or "
                         "'q_proj,v_proj,fc1,fc2' for the BART-style ones "
                         "(m2m100/mbart/nllb/seamless), confirmed "
                         "per-architecture via named_modules(). Attention "
                         "alone (the original LoRA paper's q,v setting) "
                         "was tried first and dropped: it reweights what "
                         "the model attends to but leaves the feed-forward "
                         "layers frozen, and inserting a terminology gloss "
                         "is a content-injection task, not an attention-"
                         "steering one -- confirmed directly, afriteva's "
                         "glossary accuracy collapsed from 41.7% (full "
                         "fine-tune) to 0.04% under q,v-only LoRA despite "
                         "normal BLEU/loss convergence (see "
                         "BENCHMARK_REPORT.md).")

parser.add_argument("--resume", nargs="?", const="auto", default=None,
                    metavar="CHECKPOINT",
                    help="Resume training. Bare --resume picks up the latest "
                         "checkpoint in the output dir; pass a path to choose "
                         "one. Without this, checkpoints are written but never "
                         "read, so an interrupted run restarts from zero.")

parser.add_argument("--save-strategy", default="steps", choices=["steps", "epoch"],
                    help="Step-based saving bounds what a crash costs. Epoch "
                         "boundaries are ~22min on a base model and over an "
                         "hour at 1.2B, and a crash before one loses everything.")

parser.add_argument("--save-steps", type=int, default=500,
                    help="Checkpoint interval when --save-strategy=steps. "
                         "load_best_model_at_end forces evaluation onto the "
                         "same cadence, so this is the eval interval too.")

parser.add_argument("--early-stopping-patience", type=int, default=6,
                    help="Evaluations without improvement before stopping. "
                         "At --save-steps 500 each unit is 500 steps. "
                         "Raised from 4: that cut mt5 off at step 3500/6560 "
                         "before it had learned to gloss at all (see "
                         "BENCHMARK_REPORT.md section 5); 6 let both mT5 "
                         "runs reach full convergence with no downside.")

parser.add_argument("--eval-subset", type=int, default=1000,
                    help="Evaluate on this many validation examples during "
                         "training (0 = all). Checkpoint ranking does not need "
                         "the full set, and this is what keeps frequent "
                         "evaluation affordable. Test scoring always uses "
                         "the full test split.")

args = parser.parse_args()

# ============================================================
# SELECT MODEL
# ============================================================

MODEL_KEY = args.model

config = MODEL_CONFIGS[MODEL_KEY]

MODEL_NAME = config["model_name"]

SOURCE_PREFIX = config["source_prefix"]

# Empty for the T5-family models, which steer with a text prefix instead.
LANG_STYLE = config.get("lang_style")

SOURCE_LANG = args.src_lang_code or config.get("src_lang_code")

TARGET_LANG = args.tgt_lang_code or config.get("tgt_lang_code")

TARGET_COLUMN = args.target_column

BATCH_SIZE = args.batch_size or config["batch_size"]

EFFECTIVE_BATCH = args.effective_batch or config["effective_batch"]

# LoRA freezes the base model, so there's no optimizer state for it at
# all -- Adafactor's memory saving over AdamW is moot with ~1% of params
# trainable, and AdamW's per-parameter moments are cheap at that scale.
# Forced regardless of the model config's own full-FT optim choice.
OPTIM = "adamw_torch" if args.lora else (args.optim or config.get("optim", "adamw_torch"))

# LoRA's randomly-initialized A / zero-initialized B matrices start further
# from a useful update than full fine-tuning's pretrained weights do, so it
# conventionally wants a higher rate -- 1e-3 is the standard LoRA-paper
# starting point, well above every full-FT rate in this sweep (max 1e-3
# only for the Adafactor tier, otherwise 1e-4 or below).
LORA_DEFAULT_LR = 1e-3
LEARNING_RATE = args.learning_rate or (LORA_DEFAULT_LR if args.lora else config["learning_rate"])

# Every GPU contributes BATCH_SIZE examples per step, so accumulation only
# has to make up whatever is left to reach the target global batch.
GRAD_ACCUM = max(1, EFFECTIVE_BATCH // (BATCH_SIZE * WORLD_SIZE))

# No optimizer state during generation, so eval can run wider than train.
EVAL_BATCH_SIZE = args.eval_batch_size or max(1, BATCH_SIZE * 2)

# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = args.train_file

VAL_FILE = args.val_file

TEST_FILE = args.test_file

OUTPUT_DIR = args.output_dir or os.path.join(
    DATA_DIR, f"output_{'lora_' if args.lora else ''}{MODEL_KEY}"
)

if IS_MAIN:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# TRAINING SETTINGS
# ============================================================

MAX_SOURCE_LENGTH = args.max_source_length
MAX_TARGET_LENGTH = args.max_target_length

NUM_EPOCHS = args.epochs

NUM_BEAMS = args.num_beams

# ============================================================
# TRAINING ARGS
# ============================================================
# Built before the data pipeline so main_process_first() is available
# for the download/tokenize steps.

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    do_train=True,
    do_eval=True,

    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    # Per-epoch validation only has to rank checkpoints; the final test
    # pass below re-runs generation with the full --num-beams.
    generation_num_beams=args.eval_num_beams,

    # load_best_model_at_end requires both strategies to agree, and requires
    # save_steps to be a multiple of eval_steps -- so saving cannot be more
    # frequent than evaluating. Keeping them equal is the finest granularity
    # available; --eval-subset is what makes that cadence cheap enough.
    eval_strategy=args.save_strategy,
    eval_steps=args.save_steps,

    save_strategy=args.save_strategy,
    save_steps=args.save_steps,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    gradient_accumulation_steps=GRAD_ACCUM,

    # Activations dominate the fp32 footprint on 32GB cards.
    gradient_checkpointing=not args.no_gradient_checkpointing,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Stream generated sequences to CPU instead of holding the whole
    # eval set on the GPU until the loop ends.
    eval_accumulation_steps=8,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=0.01,

    # adafactor for the 1B+ tier; fp32 AdamW does not fit there.
    optim=OPTIM,

    save_total_limit=2,

    logging_steps=50,

    load_best_model_at_end=True,

    metric_for_best_model="chrf",

    greater_is_better=True,

    report_to="none",

    seed=SEED,

    # V100 (compute capability 7.0) has no bf16; mT5 is unstable in fp16,
    # so fp32 is the only safe choice for this family.
    fp16=False,
    bf16=False,

    # ---- distributed / throughput ----
    ddp_backend="nccl",

    # True costs an extra graph traversal per step but is the only safe
    # default across architectures: M2M-100 has parameters that do not
    # participate in every forward pass (confirmed -- False crashes it with
    # "Expected to have finished reduction..."), where the T5-family models
    # tested so far do not. Override per-model if a config is confirmed safe.
    ddp_find_unused_parameters=config.get("ddp_find_unused_parameters", True),

    dataloader_num_workers=args.num_workers,
    dataloader_pin_memory=True,

    # group_by_length=args.group_by_length,
)

log("=" * 80)
log(f"MODEL           : {MODEL_KEY} ({MODEL_NAME}, {config.get('params', '?')})")
log(f"OPTIMIZER       : {OPTIM}")
log(f"WORLD SIZE      : {WORLD_SIZE} process(es)")
log(f"BATCH / GPU     : {BATCH_SIZE}")
log(f"EVAL BATCH / GPU: {EVAL_BATCH_SIZE}")
log(f"BEAMS eval/test : {args.eval_num_beams} / {NUM_BEAMS}")
log(f"TARGET COLUMN   : {TARGET_COLUMN}")
log(f"GRAD ACCUM      : {GRAD_ACCUM}")
log(f"EFFECTIVE BATCH : {BATCH_SIZE * WORLD_SIZE * GRAD_ACCUM}")
log(f"OUTPUT DIR      : {OUTPUT_DIR}")
log("=" * 80)

# ============================================================
# LOAD DATASET
# ============================================================
# main_process_first: rank 0 does the download/parse, the other ranks
# then hit the warm cache instead of racing on the same files.

with training_args.main_process_first(desc="dataset loading"):

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VAL_FILE,
            "test": TEST_FILE
        }
    )

# ============================================================
# TOKENIZER
# ============================================================

# UBC-NLP's toucan-1.2B and cheetah-1.2B both declare
# tokenizer_class: T5Tokenizer in tokenizer_config.json, but both repos'
# actual tokenizer.json is a plain BPE model, not the Unigram/
# SentencePiece format T5's fast-tokenizer conversion expects.
# AutoTokenizer forces the T5 class per that declaration and crashes
# with "TypeError: argument 'vocab': 'dict' object cannot be converted
# to 'Sequence'" -- confirmed identical on both models, with use_fast
# True and False alike, so this is a repo-level tokenizer_class/
# tokenizer.json mismatch (same export pipeline, same bug), not a local
# environment issue. tokenizer.json loads correctly on its own via the
# generic fast-tokenizer wrapper, bypassing the T5-class resolution
# entirely.
if MODEL_KEY in {"toucan", "cheetah"}:

    tokenizer_path = hf_hub_download(MODEL_NAME, "tokenizer.json")

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

# ============================================================
# MULTILINGUAL LANGUAGE SETUP
# ============================================================
# NLLB, mBART-50, M2M-100 and SeamlessM4T are steered by language tokens
# rather than a text prefix. Setting tgt_lang makes the tokenizer prepend
# the right token to the labels -- confirmed for SeamlessM4T specifically
# by direct check: tokenizing the same text_target with tgt_lang set to
# two different languages produces two different label sequences, so this
# is not a no-op for this tokenizer. forced_bos_token_id (applied to the
# model further down) makes generation start from it -- also confirmed
# directly for SeamlessM4T: despite exposing a generate(tgt_lang=...)
# kwarg (which the Trainer's own internal eval/predict generate() calls
# have no way to supply), setting only generation_config.forced_bos_token_id
# and calling generate() with no tgt_lang produces a first token exactly
# matching that id and genuinely-decoded text in the target language --
# so the existing mechanism transfers without needing Trainer changes.
# Without either of these steps the model decodes into whatever language
# it defaults to, which is the single easiest way to get silently wrong
# output from any of these four.

FORCED_BOS_TOKEN_ID = None

if LANG_STYLE in {"nllb", "mbart", "m2m100", "seamless"}:

    tokenizer.src_lang = SOURCE_LANG
    tokenizer.tgt_lang = TARGET_LANG

    # lang_code_to_id (when the tokenizer defines it) is the authoritative
    # mapping these tokenizer classes use internally for language control
    # tokens, and must be preferred over convert_tokens_to_ids. Confirmed
    # the hard way: M2M-100's real control token for "en" is the internal
    # form "__en__" at id 128022, but convert_tokens_to_ids("en") silently
    # resolves to an unrelated ordinary vocabulary entry at id 49 -- a
    # valid, non-unk id, so it passed the id != unk_token_id check below
    # and only surfaced as the literal string "en" leaking into every
    # generated sentence. mBART-50 exposes the same mapping and agrees
    # with convert_tokens_to_ids there; NLLB's tokenizer (this version)
    # does not expose lang_code_to_id at all and convert_tokens_to_ids is
    # correct for it, since its codes are themselves ordinary vocab tokens.
    lang_map = getattr(tokenizer, "lang_code_to_id", None)

    if lang_map and TARGET_LANG in lang_map:
        token_id = lang_map[TARGET_LANG]
    else:
        token_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    if token_id is None or token_id == tokenizer.unk_token_id:
        raise SystemExit(
            f"target language token {TARGET_LANG!r} is not in the "
            f"{MODEL_NAME} vocabulary. Pass a supported --tgt-lang-code."
        )

    FORCED_BOS_TOKEN_ID = token_id

    log(f"language tokens : {SOURCE_LANG} -> {TARGET_LANG} (bos id {token_id})")

# ============================================================
# PREPROCESS
# ============================================================

def preprocess_function(examples):

    # Data files are flat records: {"en": ..., "pcm": ...}
    if "translation" in examples:
        sources = [pair["en"] for pair in examples["translation"]]
        targets = [pair[TARGET_COLUMN] for pair in examples["translation"]]
    else:
        sources = examples["en"]
        # Splits regenerated by prepare_data.py carry pcm_augmented and
        # pcm_clean alongside pcm; older split files only have pcm.
        column = TARGET_COLUMN if TARGET_COLUMN in examples else "pcm"
        targets = examples[column]

    inputs = [SOURCE_PREFIX + source for source in sources]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# ============================================================
# TOKENIZE
# ============================================================

with training_args.main_process_first(desc="tokenization"):

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )

# ============================================================
# IN-TRAINING VALIDATION SET
# ============================================================
# Checkpoint ranking does not need all 2,623 dev examples, and generation
# is the expensive part of every evaluation. Shuffle before taking the
# subset: the splits inherit the source CSV's ordering, which is grouped
# by subject, so an unshuffled head would be nearly all Computer science.

eval_dataset = tokenized_dataset["validation"]

if args.eval_subset and 0 < args.eval_subset < len(eval_dataset):

    eval_dataset = eval_dataset.shuffle(seed=SEED).select(range(args.eval_subset))

    log(f"VALIDATION      : {len(eval_dataset)} of "
        f"{len(tokenized_dataset['validation'])} examples (seeded subsample)")

# ============================================================
# MODEL
# ============================================================

# SeamlessM4T v2 is a multimodal (speech+text) checkpoint; AutoModelFor
# Seq2SeqLM does not resolve to a usable class for it. The dedicated
# text-to-text class loads only the text encoder/decoder weights and
# discards the speech encoder/vocoder/T2U components as "unexpected" --
# confirmed directly: 1.37B params actually loaded (5.5GB in fp32)
# against the repo's 9.2GB full multimodal checkpoint, and a real
# forward+backward pass with labels runs cleanly (loss finite, no error).
if MODEL_KEY == "seamless":
    model = SeamlessM4Tv2ForTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

# Captured immediately after loading, before anything else touches the
# model -- confirmed directly (§3.20-class bug, found while re-scoring
# saved checkpoints) that saving through this pipeline silently flips
# tie_word_embeddings to True for every checkpoint whose real upstream
# architecture has it False (mt5/afrimt5/mt5_large/afriteva_v2_large/
# cheetah/toucan -- T5 v1.1-style genuinely untied embeddings, all 6
# affected). The saved model.safetensors still contains the correct,
# separately fine-tuned lm_head.weight in every case checked; only the
# config flag is wrong, which makes from_pretrained() discard that
# tensor and silently substitute a tied copy of the input embeddings
# instead -- producing fluent-looking garbage on every future reload,
# not a crash. Root cause in the Trainer/accelerate save stack was not
# isolated further; re-asserted directly after save below instead,
# which is correct regardless of which internal step causes the drift.
ORIGINAL_TIE_WORD_EMBEDDINGS = model.config.tie_word_embeddings

# Gradient checkpointing and the generation cache are mutually exclusive;
# the Trainer re-enables the cache for eval/generate on its own.
if training_args.gradient_checkpointing:
    model.config.use_cache = False

# Pin generation to the target language token for the language-token models.
# transformers 5.x hard-errors ("This strategy to control generation is not
# supported anymore") if forced_bos_token_id is set on model.config at all,
# even alongside a correctly-set generation_config -- confirmed by crashing
# m2m100 at its first evaluation. generation_config is the only place this
# may be set now.
if FORCED_BOS_TOKEN_ID is not None:
    model.generation_config.forced_bos_token_id = FORCED_BOS_TOKEN_ID

# ============================================================
# LoRA
# ============================================================
# Wrapped after the language-token / dtype setup above so all of that
# still applies to the frozen base model underneath. Only q/v attention
# projections get adapters by default -- the original LoRA paper's
# setting, and confirmed (named_modules() on afriteva_base) to be real
# module names across every T5-family checkpoint in this sweep.

if args.lora:

    lora_target_modules = (
        args.lora_target_modules or config.get("lora_target_modules", "q,v,wi,wo")
    ).split(",")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
    )

    model = get_peft_model(model, lora_config)

    if IS_MAIN:
        model.print_trainable_parameters()

    # Needed alongside gradient checkpointing specifically: with the base
    # model frozen, the first differentiable op sits inside a checkpointed
    # block, and autograd needs enable_input_require_grads() to keep the
    # embedding output requiring grad through it. Confirmed directly by an
    # isolated forward+backward test -- omitting this silently zeroes
    # every LoRA gradient under checkpointing rather than erroring.
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

# ============================================================
# METRICS
# ============================================================

with training_args.main_process_first(desc="metric download"):

    bleu_metric = evaluate.load("sacrebleu")

    chrf_metric = evaluate.load("chrf")

    ter_metric = evaluate.load("ter")

# ============================================================
# COMPUTE METRICS
# ============================================================

def compute_metrics(eval_preds):

    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # ── Sanitize preds ──────────────────────────────────────
    # Trainer pads prediction arrays with -100 or large sentinel
    # values; clip to valid vocab range before decoding.
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1).astype(np.int32)

    # ── Sanitize labels ─────────────────────────────────────
    labels = labels.astype(np.int32)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # ── Decode ──────────────────────────────────────────────
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds  = [pred.strip()  for pred  in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # ── BLEU ────────────────────────────────────────────────
    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[x] for x in decoded_labels]
    )

    # ── ChrF++ ──────────────────────────────────────────────
    # word_order=0 (the evaluate default) is plain chrF. chrF++ is
    # word_order=2 -- reporting the default as "chrF++" mislabels it.
    chrf = chrf_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    chrf_pp = chrf_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        word_order=2
    )

    # ── TER ─────────────────────────────────────────────────
    ter = ter_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    # ── Save predictions CSV ─────────────────────────────────
    # Only rank 0 writes, otherwise every GPU races on the same path.
    if IS_MAIN:
        csv_path = os.path.join(OUTPUT_DIR, "predictions.csv")
        pd.DataFrame({
            "reference":  decoded_labels,
            "prediction": decoded_preds
        }).to_csv(csv_path, index=False)
        print(f"Saved predictions -> {csv_path}", flush=True)

    # ── Result ───────────────────────────────────────────────
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id)
        for pred in preds
    ]

    return {
        "bleu":    round(bleu["score"],  4),
        "chrf":    round(chrf["score"],    4),
        "chrf++":  round(chrf_pp["score"], 4),
        "ter":     round(ter["score"],   4),
        "gen_len": round(np.mean(prediction_lens), 4),
    }

# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

# ============================================================
# TRAINER
# ============================================================
# The Trainer reads LOCAL_RANK from the environment and wraps the model
# in DistributedDataParallel itself, so nothing here is rank-specific.

trainer = Seq2SeqTrainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_dataset["train"],

    eval_dataset=eval_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            # Counted in EVALUATIONS, not epochs. At --save-steps 500 that
            # is 500 steps per unit, so the epoch-era value of 2 would now
            # stop a run after 1000 steps of no improvement -- far too eager.
            early_stopping_patience=args.early_stopping_patience
        )
    ]
)

# ============================================================
# TRAIN
# ============================================================

log("=" * 80)
log(f"STARTING TRAINING: {MODEL_KEY}")
log("=" * 80)

# True lets the Trainer find the newest checkpoint in output_dir itself;
# a string resumes from that exact path; None starts fresh.
RESUME_FROM = True if args.resume == "auto" else args.resume

if RESUME_FROM:
    log(f"RESUMING from {RESUME_FROM if isinstance(RESUME_FROM, str) else 'latest checkpoint'}")

trainer.train(resume_from_checkpoint=RESUME_FROM)

# ============================================================
# TEST EVALUATION
# ============================================================

log("=" * 80)
log("RUNNING TEST EVALUATION")
log("=" * 80)

log(f"generating with {NUM_BEAMS} beams")

results = trainer.predict(
    tokenized_dataset["test"],
    num_beams=NUM_BEAMS,
    max_length=MAX_TARGET_LENGTH,
)

metrics = results.metrics

log(metrics)

# ============================================================
# SAVE MODEL
# ============================================================
# save_model is rank-aware internally; the tokenizer is not.

trainer.save_model(OUTPUT_DIR)

if trainer.is_world_process_zero():

    tokenizer.save_pretrained(OUTPUT_DIR)

    # Guard against the tie_word_embeddings corruption documented above --
    # patched directly in the saved config.json rather than trusted to
    # have been preserved through the save, since the drift happens
    # somewhere inside the Trainer/accelerate save path itself.
    if not args.lora:
        config_path = os.path.join(OUTPUT_DIR, "config.json")
        with open(config_path) as handle:
            saved_config = json.load(handle)
        if saved_config.get("tie_word_embeddings") != ORIGINAL_TIE_WORD_EMBEDDINGS:
            log(f"WARNING: tie_word_embeddings drifted during save "
                f"({saved_config.get('tie_word_embeddings')} != "
                f"{ORIGINAL_TIE_WORD_EMBEDDINGS}) -- correcting in "
                f"{config_path}")
            saved_config["tie_word_embeddings"] = ORIGINAL_TIE_WORD_EMBEDDINGS
            with open(config_path, "w") as handle:
                json.dump(saved_config, handle, indent=2)

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "metrics.json"
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics -> {metrics_path}", flush=True)

    # ========================================================
    # SAVE TEST PREDICTIONS
    # ========================================================

    # Generated sequences are right-padded with -100, which batch_decode
    # cannot handle; swap those back to the pad token first.
    test_preds = results.predictions

    if isinstance(test_preds, tuple):
        test_preds = test_preds[0]

    test_preds = np.where(test_preds != -100, test_preds, tokenizer.pad_token_id)

    predictions = tokenizer.batch_decode(
        test_preds,
        skip_special_tokens=True
    )

    test_records = list(dataset["test"])

    # Under DDP the distributed sampler pads the split so it divides evenly
    # across ranks, so predict() can return a few more rows than the dataset
    # has (2623 examples at 16/step -> 2624). The padding is appended at the
    # end, so trimming to the record count realigns them.
    if len(predictions) != len(test_records):

        print(f"aligning {len(predictions)} predictions to "
              f"{len(test_records)} test records", flush=True)

        predictions = predictions[:len(test_records)]

    # Anything shorter would mean predictions and records no longer line up,
    # which would silently mis-pair every row after the gap.
    if len(predictions) < len(test_records):
        raise SystemExit(
            f"only {len(predictions)} predictions for "
            f"{len(test_records)} test records; refusing to write a "
            f"misaligned predictions file."
        )

    def column(name, fallback=None):
        return [
            record.get(name, record.get(fallback, ""))
            for record in test_records
        ]

    prediction_df = pd.DataFrame({
        # Carried through so glossary_metrics.py can join on id rather
        # than relying on row order.
        "id":         column("id"),
        "source":     column("en"),
        "reference":  column(TARGET_COLUMN, "pcm"),
        "prediction": predictions,
    })

    prediction_csv = os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_csv,
        index=False
    )

    print(f"Saved predictions -> {prediction_csv}", flush=True)

    # ========================================================
    # GLOSSARY EVALUATION
    # ========================================================
    # Dual-reference scoring: MT metrics against clean references so the
    # numbers stay comparable to other English-Pidgin work, plus the
    # terminology metrics that the glossary augmentation is actually for.
    # Skipped rather than fatal when the splits predate prepare_data.py.

    if test_records and "glossed_terms" in test_records[0]:

        from glossary_metrics import build_term_inventory, evaluate_predictions

        inventory = build_term_inventory(
            list(dataset["train"]),
            list(dataset["validation"]),
            test_records,
        )

        report, per_term_rows = evaluate_predictions(
            predictions, test_records, inventory
        )

        report["model"] = MODEL_KEY
        report["model_name"] = MODEL_NAME
        report["target_column"] = TARGET_COLUMN

        report_path = os.path.join(OUTPUT_DIR, "glossary_report.json")

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        pd.DataFrame(per_term_rows).to_csv(
            os.path.join(OUTPUT_DIR, "per_term_glosses.csv"), index=False
        )

        glossary = report["glossary"]

        print(f"Saved glossary report -> {report_path}", flush=True)
        print(f"  BLEU (clean refs)   : {report['clean_reference']['bleu']}", flush=True)
        print(f"  chrF++ (clean refs) : {report['clean_reference']['chrf++']}", flush=True)
        print(f"  gloss accuracy      : {glossary['gloss_accuracy']}", flush=True)
        print(f"  gloss F1            : {glossary['gloss_f1']}", flush=True)

    else:
        print("Splits lack glossary metadata; run prepare_data.py to enable "
              "glossary scoring.", flush=True)

log("=" * 80)
log("TRAINING COMPLETE")
log("=" * 80)

# ============================================================
# CLEANUP
# ============================================================

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Tear the process group down cleanly so torchrun does not report
# a stale-rank warning on exit.
if dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()
