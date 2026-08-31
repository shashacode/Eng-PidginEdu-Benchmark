# ============================================================
# AFRICOMET SCORING FOR ENGLISH -> NIGERIAN PIDGIN
# ============================================================
# Reference-based neural MT quality metric, trained specifically for
# African languages including Nigerian Pidgin (pcm is in its 76-language
# list -- checked directly against the model card before using it).
#
#     masakhane/africomet-stl-1.1
#     https://arxiv.org/abs/2311.09828 (Wang et al., 2023)
#     Apache-2.0, not gated
#
# Scored against CLEAN references (glosses stripped from both prediction
# and reference), the same way BLEU/chrF++ are scored for cross-work
# comparability (glossary_metrics.py's dual-reference design). AfriCOMET
# was trained on natural parallel sentences, not text containing inline
# parenthetical glosses -- scoring it against the glossary-augmented
# form would push every example out of its training distribution for no
# benefit, since gloss quality is already measured separately by the
# glossary-accuracy metric.
#
# Usage:
#     python africomet_metrics.py --predictions output_afrimt5/test_predictions.csv
#     python africomet_metrics.py --model-dir output_afrimt5
#     python africomet_metrics.py --all               # every output_*/ with predictions
# ============================================================

import os
import json
import glob
import argparse

import pandas as pd

from glossary_metrics import build_term_inventory, strip_glosses

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

AFRICOMET_MODEL = "masakhane/africomet-stl-1.1"

_model = None


def get_model():
    """Lazy singleton: the checkpoint load (~1-2GB) only happens once
    even when scoring many models in one process."""

    global _model

    if _model is None:
        from comet import download_model, load_from_checkpoint

        model_path = download_model(AFRICOMET_MODEL)
        _model = load_from_checkpoint(model_path)

    return _model


def score_triples(sources, predictions, references, batch_size=16, gpus=1):
    """
    sources/predictions/references: parallel lists of strings, already
    clean (gloss-stripped) -- this function does not strip glosses itself,
    callers are expected to have done that (see evaluate_predictions below
    for the version that does).

    Returns (system_score, per_segment_scores).
    """

    model = get_model()

    data = [
        {"src": s, "mt": m, "ref": r}
        for s, m, r in zip(sources, predictions, references)
    ]

    output = model.predict(data, batch_size=batch_size, gpus=gpus)

    return output.system_score, list(output.scores)


def evaluate_predictions(sources, predictions, references, inventory,
                          batch_size=16, gpus=1):
    """
    Strips glosses from predictions and references, then scores.
    references here should be the augmented form (pcm_augmented) --
    stripping happens inside this function, matching glossary_metrics.py's
    clean_reference path so the two modules can't drift apart on what
    "clean" means.
    """

    clean_predictions = [strip_glosses(p, inventory) for p in predictions]
    clean_references = [strip_glosses(r, inventory) for r in references]

    system_score, segment_scores = score_triples(
        sources, clean_predictions, clean_references,
        batch_size=batch_size, gpus=gpus,
    )

    return {
        "africomet_stl": round(system_score * 100, 4),  # 0-1 -> 0-100, matches BLEU/chrF scale
        "africomet_stl_raw": round(system_score, 6),     # untransformed, for anyone who wants it
        "n_examples": len(predictions),
        "model": AFRICOMET_MODEL,
    }, segment_scores


# ============================================================
# CLI
# ============================================================

def load_split(data_dir, split):

    path = os.path.join(data_dir, f"{split}.json")

    with open(path) as handle:
        return json.load(handle)


def score_one(predictions_csv, data_dir, split, inventory, batch_size, gpus,
              per_segment_out=None):

    frame = pd.read_csv(predictions_csv)

    if "prediction" not in frame.columns:
        raise SystemExit(f"{predictions_csv} has no 'prediction' column")

    predictions = frame["prediction"].fillna("").astype(str).tolist()
    sources = frame["source"].fillna("").astype(str).tolist()

    # test_predictions.csv already carries clean-form "reference"? No --
    # both train.py and glossary_metrics.py write the augmented text under
    # "reference"; strip it the same way clean_reference scoring does.
    references = frame["reference"].fillna("").astype(str).tolist()

    report, segment_scores = evaluate_predictions(
        sources, predictions, references, inventory,
        batch_size=batch_size, gpus=gpus,
    )

    if per_segment_out:
        out = frame.copy()
        out["africomet"] = segment_scores
        out.to_csv(per_segment_out, index=False)

    return report


def main():

    parser = argparse.ArgumentParser(
        description="Score predictions with AfriCOMET (clean references)."
    )

    parser.add_argument("--predictions", default=None,
                        help="A single test_predictions.csv to score.")
    parser.add_argument("--model-dir", default=None,
                        help="output_<model>/ directory containing test_predictions.csv; "
                             "writes africomet_report.json into it.")
    parser.add_argument("--all", action="store_true",
                        help="Score every output_*/test_predictions.csv found, "
                             "writing africomet_report.json into each.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--per-term-detail", action="store_true",
                        help="Also write a per-example CSV with an africomet column.")

    args = parser.parse_args()

    inventory = build_term_inventory(
        *[load_split(args.data_dir, s) for s in ("train", "dev", "test")]
    )

    targets = []

    if args.predictions:
        targets.append((args.predictions, None))
    elif args.model_dir:
        targets.append((
            os.path.join(args.model_dir, "test_predictions.csv"),
            args.model_dir,
        ))
    elif args.all:
        for csv_path in sorted(glob.glob(
            os.path.join(args.data_dir, "output_*", "test_predictions.csv")
        )):
            targets.append((csv_path, os.path.dirname(csv_path)))
    else:
        raise SystemExit("pass one of --predictions, --model-dir, --all")

    for csv_path, model_dir in targets:

        if not os.path.exists(csv_path):
            print(f"skip (no predictions): {csv_path}")
            continue

        name = os.path.basename(model_dir) if model_dir else csv_path

        print(f"=== {name} ===")

        per_segment_out = (
            os.path.join(model_dir, "per_example_africomet.csv")
            if (model_dir and args.per_term_detail) else None
        )

        report = score_one(
            csv_path, args.data_dir, args.split, inventory,
            args.batch_size, args.gpus, per_segment_out,
        )

        print(f"  africomet_stl : {report['africomet_stl']}")
        print(f"  n_examples    : {report['n_examples']}")

        if model_dir:
            out_path = os.path.join(model_dir, "africomet_report.json")
            with open(out_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"  wrote -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
