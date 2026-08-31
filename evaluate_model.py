# ============================================================
# STANDALONE MODEL EVALUATION -- fine-tuned re-scoring AND zero-shot
# ============================================================
# Two uses:
#
#   1. Re-score an already fine-tuned model directory without retraining
#      it (train.py does this automatically at the end of a run, but a
#      crash after training, or a change to the metric, should never
#      cost a retrain).
#
#   2. Score a model's ZERO-SHOT performance -- the original, un-fine-
#      tuned checkpoint, before any Pidgin exposure. This is the
#      baseline the "zero-shot vs fine-tuned" research questions (see
#      BENCHMARK_REPORT.md §9.5) are built on.
#
# The per-model loading logic (tokenizer special-cases for toucan/
# cheetah's broken tokenizer_class, §3.16; the language-token mechanism
# for nllb/mbart/m2m100/madlad/seamless, §3.12/§9.2; SeamlessM4T's
# dedicated model class, §9.2) mirrors train.py's MODEL_CONFIGS and
# loading code as closely as possible -- duplicated here rather than
# imported, since train.py is a top-level script with side effects on
# import (argparse, GPU init, dataset loading), not a library. Kept in
# sync by hand; if train.py's per-model handling changes, this file
# needs the matching update. Only the fields this script actually needs
# are carried (source_prefix, lang_style, src/tgt_lang_code) --
# training-only fields (batch_size, learning_rate, optim) are omitted.
#
# Zero-shot output goes to a SEPARATE directory (output_zeroshot_<key>/)
# rather than overwriting output_<key>/ -- confirmed necessary: the
# original version of this script wrote into --model-dir directly, which
# for a fine-tuned model's own directory is correct (re-scoring), but
# for zero-shot would have silently clobbered the fine-tuned run's own
# test_predictions.csv/glossary_report.json if the same directory
# naming were reused by mistake.
#
# Usage:
#     # re-score an already fine-tuned model
#     python evaluate_model.py --model-key afrimt5 --model-dir output_afrimt5
#
#     # zero-shot: the original checkpoint, never fine-tuned
#     python evaluate_model.py --model-key afrimt5 --zero-shot
#     python evaluate_model.py --model-key toucan --zero-shot --num-beams 5
# ============================================================

import os
import json
import argparse

import torch
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast,
    SeamlessM4Tv2ForTextToText,
)

from huggingface_hub import hf_hub_download

from glossary_metrics import build_term_inventory, evaluate_predictions

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# Mirrors train.py's MODEL_CONFIGS -- only the fields needed to load a
# model and steer generation correctly. madlad3b and t5v11xl are
# included here even though they were dropped from full fine-tuning
# (BENCHMARK_REPORT.md §9.3, memory ceiling) -- that ceiling is a
# training-time (gradient + optimizer state) problem, not an inference
# one, so zero-shot evaluation of both is still meaningful and cheap.
# ------------------------------------------------------------

MODEL_REGISTRY = {
    "afriteva": {
        "model_name": "castorini/afriteva_base",
        "source_prefix": "translate English to Pidgin: ",
    },
    "m2m100": {
        "model_name": "facebook/m2m100_418M",
        "source_prefix": "",
        "lang_style": "m2m100",
        "src_lang_code": "en",
        "tgt_lang_code": "en",
    },
    "m2m100_1.2b": {
        "model_name": "facebook/m2m100_1.2B",
        "source_prefix": "",
        "lang_style": "m2m100",
        "src_lang_code": "en",
        "tgt_lang_code": "en",
    },
    "mt5": {
        "model_name": "google/mt5-base",
        "source_prefix": "translate English to Pidgin: ",
    },
    "afrimt5": {
        "model_name": "masakhane/afri-mt5-base",
        "source_prefix": "translate English to Pidgin: ",
    },
    "nllb": {
        "model_name": "facebook/nllb-200-distilled-600M",
        "source_prefix": "",
        "lang_style": "nllb",
        "src_lang_code": "eng_Latn",
        "tgt_lang_code": "tpi_Latn",
    },
    "mbart50": {
        "model_name": "facebook/mbart-large-50-many-to-many-mmt",
        "source_prefix": "",
        "lang_style": "mbart",
        "src_lang_code": "en_XX",
        "tgt_lang_code": "en_XX",
    },
    "afriteva_v2_large": {
        "model_name": "castorini/afriteva_v2_large",
        "source_prefix": "translate English to Pidgin: ",
    },
    "mt5_large": {
        "model_name": "google/mt5-large",
        "source_prefix": "translate English to Pidgin: ",
    },
    "cheetah": {
        "model_name": "UBC-NLP/cheetah-1.2B",
        "source_prefix": "translate English to Pidgin: ",
        "custom_tokenizer": True,   # §3.16
    },
    "toucan": {
        "model_name": "UBC-NLP/toucan-1.2B",
        # Zero-shot uses the ORIGINAL "<2pcm> " prefix, not the
        # "translate English to Pidgin: " the fine-tuned run switched to
        # after §3.17 -- that switch was a fine-tuning-time fix (the
        # fragmented-token prefix was a likely contributor to the
        # training failure), but zero-shot is testing this checkpoint's
        # own native convention, and "<2pcm> " is what toucan actually
        # responded to zero-shot when this was first checked (§3.16's
        # pre-launch sanity test: coherent, on-topic Pidgin output).
        "source_prefix": "<2pcm> ",
        "custom_tokenizer": True,   # §3.16
    },
    "seamless": {
        "model_name": "facebook/seamless-m4t-v2-large",
        "source_prefix": "",
        "lang_style": "seamless",
        "src_lang_code": "__eng__",
        "tgt_lang_code": "__eng__",
        "seamless_class": True,     # §9.2
    },
    "madlad3b": {
        "model_name": "google/madlad400-3b-mt",
        "source_prefix": "<2kri> ",   # §9.3 -- verified atomic token
    },
    "t5v11xl": {
        "model_name": "google/t5-v1_1-xl",
        "source_prefix": "translate English to Pidgin: ",
    },
}


def resolve_target_lang_token(tokenizer, target_lang):
    """
    Mirrors train.py's resolution: prefer lang_code_to_id (the
    authoritative mapping for M2M-100/mBART-50) over convert_tokens_to_ids,
    which silently resolved M2M-100's "en" to the wrong, ordinary vocab
    entry (§3.11) rather than the real control token.
    """

    lang_map = getattr(tokenizer, "lang_code_to_id", None)

    if lang_map and target_lang in lang_map:
        return lang_map[target_lang]

    return tokenizer.convert_tokens_to_ids(target_lang)


def load_tokenizer_and_model(model_key, config):

    model_name = config["model_name"]

    if config.get("custom_tokenizer"):
        # §3.16: toucan/cheetah's tokenizer_config.json declares
        # T5Tokenizer, but the actual tokenizer.json is plain BPE --
        # AutoTokenizer crashes trying to force the T5/Unigram
        # conversion path. Load tokenizer.json directly instead.
        tokenizer_path = hf_hub_download(model_name, "tokenizer.json")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            pad_token="<pad>",
            eos_token="</s>",
            unk_token="<unk>",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    forced_bos_token_id = None

    lang_style = config.get("lang_style")

    if lang_style in {"nllb", "mbart", "m2m100", "seamless"}:

        tokenizer.src_lang = config["src_lang_code"]
        tokenizer.tgt_lang = config["tgt_lang_code"]

        token_id = resolve_target_lang_token(tokenizer, config["tgt_lang_code"])

        if token_id is None or token_id == tokenizer.unk_token_id:
            raise SystemExit(
                f"target language token {config['tgt_lang_code']!r} not "
                f"found in {model_name}'s vocabulary."
            )

        forced_bos_token_id = token_id

    if config.get("seamless_class"):
        # §9.2: SeamlessM4Tv2Model (what AutoModelForSeq2SeqLM resolves
        # to) is the full multimodal checkpoint; the text-to-text class
        # loads only the relevant encoder/decoder weights.
        model = SeamlessM4Tv2ForTextToText.from_pretrained(
            model_name, dtype=torch.float32
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=torch.float32
        )

    if forced_bos_token_id is not None:
        model.generation_config.forced_bos_token_id = forced_bos_token_id

    return tokenizer, model


def main():

    parser = argparse.ArgumentParser(
        description="Score a model on the test split -- fine-tuned re-score or zero-shot."
    )

    parser.add_argument("--model-key", required=True, choices=sorted(MODEL_REGISTRY),
                        help="Which model config to use (matches train.py's MODEL_CONFIGS keys).")
    parser.add_argument("--model-dir", default=None,
                        help="Local fine-tuned checkpoint directory to re-score. "
                             "Omit and pass --zero-shot instead to score the base checkpoint.")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Score the original HF checkpoint (config's model_name), "
                             "never fine-tuned. Writes to output_zeroshot_<model-key>/.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--source-prefix", default=None,
                        help="Overrides the prefix from MODEL_REGISTRY.")
    parser.add_argument("--gloss-threshold", type=float, default=50.0)
    parser.add_argument("--out-dir", default=None,
                        help="Where to write predictions/report. Defaults to "
                             "--model-dir, which is safe for --split test "
                             "(the usual case) but would silently overwrite "
                             "the test-set results if scoring a different "
                             "split against the same checkpoint -- pass this "
                             "explicitly whenever --split is not 'test'.")

    args = parser.parse_args()

    if bool(args.model_dir) == bool(args.zero_shot):
        raise SystemExit("pass exactly one of --model-dir or --zero-shot")

    config = MODEL_REGISTRY[args.model_key]

    if args.zero_shot:
        load_from = config["model_name"]
        out_dir = os.path.join(args.data_dir, f"output_zeroshot_{args.model_key}")
        os.makedirs(out_dir, exist_ok=True)
    else:
        load_from = args.model_dir if os.path.isabs(args.model_dir) else \
            os.path.join(args.data_dir, args.model_dir)
        out_dir = args.out_dir or load_from
        os.makedirs(out_dir, exist_ok=True)

    prefix = args.source_prefix if args.source_prefix is not None else config["source_prefix"]

    # ------------------------------------------------------------
    # Data
    # ------------------------------------------------------------

    with open(os.path.join(args.data_dir, f"{args.split}.json")) as handle:
        records = json.load(handle)

    if records and "glossed_terms" not in records[0]:
        raise SystemExit("splits lack glossary metadata; run prepare_data.py first.")

    sources = [prefix + r["en"] for r in records]

    print(f"model-key : {args.model_key}")
    print(f"mode      : {'zero-shot' if args.zero_shot else 're-score (fine-tuned)'}")
    print(f"loading   : {load_from}")
    print(f"prefix    : {prefix!r}")
    print(f"split     : {args.split} ({len(records)} examples)")
    print(f"beams     : {args.num_beams}")

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    if args.zero_shot:
        tokenizer, model = load_tokenizer_and_model(args.model_key, config)
    else:
        # Re-scoring a fine-tuned checkpoint. seamless and the language-
        # token models save a standard AutoTokenizer-loadable checkpoint.
        # toucan/cheetah do not: their saved tokenizer_config.json still
        # declares tokenizer_class "TokenizersBackend" (§3.16 -- the same
        # bogus declaration as the original upstream checkpoint), which
        # AutoTokenizer.from_pretrained() cannot resolve -- confirmed
        # directly, this previously untested path crashed on both models
        # the first time it was actually exercised. Load the checkpoint's
        # own saved tokenizer.json directly instead, same fix as
        # load_tokenizer_and_model() above but pointed at the local
        # fine-tuned directory rather than re-downloading from the hub,
        # since the local file is what training/eval actually used.
        if config.get("custom_tokenizer"):
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=os.path.join(load_from, "tokenizer.json"),
                pad_token="<pad>",
                eos_token="</s>",
                unk_token="<unk>",
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(load_from)
        model = AutoModelForSeq2SeqLM.from_pretrained(load_from, dtype=torch.float32)

        lang_style = config.get("lang_style")
        if lang_style in {"nllb", "mbart", "m2m100", "seamless"}:
            tokenizer.src_lang = config["src_lang_code"]
            tokenizer.tgt_lang = config["tgt_lang_code"]
            token_id = resolve_target_lang_token(tokenizer, config["tgt_lang_code"])
            model.generation_config.forced_bos_token_id = token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device).eval()

    # ------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------

    predictions = []

    for start in range(0, len(sources), args.batch_size):

        batch = sources[start:start + args.batch_size]

        inputs = tokenizer(
            batch,
            max_length=args.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # The custom PreTrainedTokenizerFast built for toucan/cheetah
        # (§3.16) emits token_type_ids, which T5-style generate() rejects
        # outright ("model_kwargs ... not used by the model"). Training
        # never hit this because DataCollatorForSeq2Seq/Trainer filter
        # inputs; this manual generation loop does not, so filter here
        # to only what generate() actually needs instead of assuming the
        # tokenizer's output is already exactly right.
        gen_inputs = {
            k: v for k, v in inputs.items()
            if k in ("input_ids", "attention_mask")
        }

        with torch.no_grad():
            generated = model.generate(
                **gen_inputs,
                max_length=args.max_length,
                num_beams=args.num_beams,
            )

        predictions.extend(
            tokenizer.batch_decode(generated, skip_special_tokens=True)
        )

        done = min(start + args.batch_size, len(sources))
        print(f"\r  {done}/{len(sources)}", end="", flush=True)

    print()

    if len(predictions) != len(records):
        raise SystemExit(
            f"{len(predictions)} predictions for {len(records)} records"
        )

    # ------------------------------------------------------------
    # Score
    # ------------------------------------------------------------

    inventory = build_term_inventory(
        *[json.load(open(os.path.join(args.data_dir, f"{s}.json")))
          for s in ("train", "dev", "test")]
    )

    report, per_term_rows = evaluate_predictions(
        predictions, records, inventory, threshold=args.gloss_threshold
    )

    report["model"] = args.model_key
    report["model_name"] = config["model_name"]
    report["num_beams"] = args.num_beams
    report["scored_by"] = "evaluate_model.py"
    report["zero_shot"] = bool(args.zero_shot)

    # ------------------------------------------------------------
    # Write
    # ------------------------------------------------------------

    pd.DataFrame({
        "id":         [r["id"] for r in records],
        "source":     [r["en"] for r in records],
        "reference":  [r["pcm_augmented"] for r in records],
        "prediction": predictions,
    }).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    with open(os.path.join(out_dir, "glossary_report.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    pd.DataFrame(per_term_rows).to_csv(
        os.path.join(out_dir, "per_term_glosses.csv"), index=False
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    glossary = report["glossary"]

    print()
    print("MT metrics (clean references, glosses stripped)")
    for key, value in report["clean_reference"].items():
        print(f"  {key:<8}: {value:>9}")

    print("\nMT metrics (augmented references)")
    for key, value in report["augmented_reference"].items():
        print(f"  {key:<8}: {value:>9}")

    print("\nGlossary metrics")
    print(f"  gloss accuracy      : {glossary['gloss_accuracy']:>9}")
    print(f"  gloss presence rate : {glossary['gloss_presence_rate']:>9}")
    print(f"  gloss precision     : {glossary['gloss_precision']:>9}")
    print(f"  gloss F1            : {glossary['gloss_f1']:>9}")
    print(f"  over-glossing rate  : {glossary['over_glossing_rate']:>9}")

    print(f"\nwrote artefacts -> {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
