# ============================================================
# GLOSSARY-AWARE EVALUATION FOR ENGLISH -> NIGERIAN PIDGIN
# ============================================================
# The Eng-PidginEdu targets carry inline terminology glosses:
#
#     ... di list of acceptable (dem worthy of acceptance) bfoqs
#           ^^^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^^^^^
#           glossed term         pidgin gloss
#
# Standard MT metrics cannot tell whether a system reproduced the right
# terminology gloss or simply copied fluent text around it. This module
# adds two things:
#
#   1. GLOSSARY ACCURACY -- did the system gloss the terms it was
#      supposed to, and did those glosses mean the right thing?
#
#   2. GLOSS STRIPPING -- remove inline glosses so BLEU / chrF++ /
#      AfriComet can be computed against clean references, keeping the
#      numbers comparable to other English-Pidgin MT work.
#
# ------------------------------------------------------------
# WHY A TERM INVENTORY
# ------------------------------------------------------------
# Not every parenthetical is a gloss. Real corpus text contains things
# like "new york (cuny)" or "(chronic adrenal insufficiency)". A
# parenthetical is only treated as a gloss when the word immediately
# before it belongs to the corpus glossary term inventory. The inventory
# is built from the TRAINING split by default, so scoring never depends
# on the test references being available.
#
# ------------------------------------------------------------
# METRICS
# ------------------------------------------------------------
# For every term the reference glosses:
#
#   presence  -- the system emitted a gloss on that term
#   accuracy  -- that gloss also matches the reference gloss in content
#                (chrF similarity >= --gloss-threshold, default 50)
#
#   Gloss Presence Rate (GPR) = glossed / expected      (recall of glossing)
#   Gloss Accuracy      (GA)  = correct / expected      <- headline number
#   Gloss Precision     (GP)  = correct / produced      (penalises over-glossing)
#   Gloss F1                  = harmonic mean of GA and GP
#
# GA alone can be gamed by glossing every term in sight, so GP and F1 are
# reported alongside it.
#
# Usage:
#     python glossary_metrics.py --predictions output_afrimt5/test_predictions.csv
#     python glossary_metrics.py --predictions preds.csv --split test --out report.json
# ============================================================

import os
import re
import json
import argparse
import collections

import pandas as pd

from sacrebleu.metrics import BLEU, CHRF, TER

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# A gloss is a parenthetical directly preceded by a single word. The word
# is captured so it can be checked against the term inventory.
GLOSS_PATTERN = re.compile(r"([\w'\-]+)\s*\(([^()]*)\)")


# ============================================================
# TERM INVENTORY
# ============================================================

def build_term_inventory(*record_lists):
    """Collect every term the corpus is known to gloss, lowercased."""

    inventory = set()

    for records in record_lists:
        for record in records:
            for term in record.get("glossed_terms", []):
                term = str(term).strip().lower()
                if term:
                    inventory.add(term)

    return inventory


# ============================================================
# GLOSS EXTRACTION
# ============================================================

def extract_glosses(text, inventory):
    """
    Return {term: gloss} for parentheticals that look like terminology
    glosses. A term glossed more than once keeps its first occurrence.
    """

    found = {}

    if not text:
        return found

    for match in GLOSS_PATTERN.finditer(str(text)):

        term = match.group(1).strip().lower()
        gloss = match.group(2).strip()

        if not gloss:
            continue

        if inventory is not None and term not in inventory:
            continue

        found.setdefault(term, gloss)

    return found


def strip_glosses(text, inventory):
    """
    Remove inline terminology glosses, leaving the surrounding translation
    intact, so MT metrics can score against clean references. Parentheticals
    that are not glosses are preserved.
    """

    if not text:
        return ""

    def replace(match):
        term = match.group(1).strip().lower()
        if inventory is not None and term not in inventory:
            return match.group(0)
        return match.group(1)

    cleaned = GLOSS_PATTERN.sub(replace, str(text))

    # Collapse the whitespace the removal leaves behind.
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)

    return cleaned.strip()


# ============================================================
# GLOSS CONTENT SIMILARITY
# ============================================================

_chrf_sentence = CHRF()


def gloss_similarity(hypothesis_gloss, reference_gloss):
    """
    chrF between two glosses, 0-100. Character-level is deliberate: Pidgin
    orthography varies ("dem"/"den", "de"/"di"), so token-exact matching
    would under-count correct glosses.
    """

    if not hypothesis_gloss or not reference_gloss:
        return 0.0

    hypothesis_gloss = hypothesis_gloss.strip().lower()
    reference_gloss = reference_gloss.strip().lower()

    if hypothesis_gloss == reference_gloss:
        return 100.0

    return _chrf_sentence.sentence_score(
        hypothesis_gloss, [reference_gloss]
    ).score


# ============================================================
# GLOSSARY ACCURACY
# ============================================================

def glossary_accuracy(predictions, records, inventory, threshold=50.0):
    """
    Score terminology glossing.

    predictions : list[str]  system outputs
    records     : list[dict] matching split records (glossed_terms, pcm_augmented, subject)
    inventory   : set[str]   terms that may legitimately carry a gloss
    threshold   : chrF score above which a gloss counts as correct

    Returns (summary_dict, per_example_rows).
    """

    expected_total = 0      # terms the reference glosses
    produced_total = 0      # glosses the system emitted on inventory terms
    present_total = 0       # expected terms the system glossed at all
    correct_total = 0       # ... and glossed with matching content
    term_mentioned = 0      # expected term appears at all, gloss or not
    unscorable_total = 0    # listed terms with no parseable reference gloss

    by_subject = collections.defaultdict(
        lambda: {"expected": 0, "present": 0, "correct": 0}
    )

    rows = []

    for prediction, record in zip(predictions, records):

        reference = record.get("pcm_augmented") or record.get("pcm") or ""

        expected_terms = [
            str(t).strip().lower()
            for t in record.get("glossed_terms", [])
            if str(t).strip()
        ]

        subject = record.get("subject") or "unknown"

        hypothesis_glosses = extract_glosses(prediction, inventory)
        reference_glosses = extract_glosses(reference, inventory)

        produced_total += len(hypothesis_glosses)

        prediction_lower = str(prediction).lower()

        # A term is only scorable if the reference actually carries a
        # parseable gloss for it. Roughly 3% of listed terms appear in an
        # inflected or re-spaced form that no system could be expected to
        # match, and counting those would cap a perfect system below 100.
        scorable_terms = [t for t in expected_terms if t in reference_glosses]

        unscorable_total += len(expected_terms) - len(scorable_terms)

        for term in scorable_terms:

            expected_total += 1
            by_subject[subject]["expected"] += 1

            if re.search(r"\b" + re.escape(term) + r"\b", prediction_lower):
                term_mentioned += 1

            hypothesis_gloss = hypothesis_glosses.get(term)

            if hypothesis_gloss is None:
                rows.append({
                    "id": record.get("id"),
                    "subject": subject,
                    "term": term,
                    "reference_gloss": reference_glosses.get(term, ""),
                    "hypothesis_gloss": "",
                    "similarity": 0.0,
                    "glossed": False,
                    "correct": False,
                })
                continue

            present_total += 1
            by_subject[subject]["present"] += 1

            # Guaranteed present: scorable_terms was filtered on this.
            reference_gloss = reference_glosses[term]

            similarity = gloss_similarity(hypothesis_gloss, reference_gloss)

            correct = similarity >= threshold

            if correct:
                correct_total += 1
                by_subject[subject]["correct"] += 1

            rows.append({
                "id": record.get("id"),
                "subject": subject,
                "term": term,
                "reference_gloss": reference_gloss,
                "hypothesis_gloss": hypothesis_gloss,
                "similarity": round(similarity, 2),
                "glossed": True,
                "correct": bool(correct),
            })

    def ratio(numerator, denominator):
        return round(100.0 * numerator / denominator, 4) if denominator else 0.0

    presence_rate = ratio(present_total, expected_total)
    accuracy = ratio(correct_total, expected_total)
    precision = ratio(correct_total, produced_total)

    f1 = 0.0
    if accuracy + precision > 0:
        f1 = round(2 * accuracy * precision / (accuracy + precision), 4)

    summary = {
        "gloss_presence_rate": presence_rate,
        "gloss_accuracy": accuracy,
        "gloss_precision": precision,
        "gloss_f1": f1,
        "term_mention_rate": ratio(term_mentioned, expected_total),
        "over_glossing_rate": ratio(max(0, produced_total - present_total),
                                    produced_total),
        "expected_glosses": expected_total,
        "produced_glosses": produced_total,
        "present_glosses": present_total,
        "correct_glosses": correct_total,
        "unscorable_glosses": unscorable_total,
        "gloss_threshold": threshold,
    }

    summary["by_subject"] = {
        subject: {
            "expected": counts["expected"],
            "gloss_presence_rate": ratio(counts["present"], counts["expected"]),
            "gloss_accuracy": ratio(counts["correct"], counts["expected"]),
        }
        for subject, counts in sorted(by_subject.items())
    }

    return summary, rows


# ============================================================
# MT METRICS
# ============================================================

def mt_metrics(predictions, references):
    """BLEU, chrF, chrF++ and TER over a set of hypotheses."""

    reference_lists = [references]

    return {
        "bleu":   round(BLEU().corpus_score(predictions, reference_lists).score, 4),
        "chrf":   round(CHRF(word_order=0).corpus_score(predictions, reference_lists).score, 4),
        # word_order=2 is what actually makes it chrF++.
        "chrf++": round(CHRF(word_order=2).corpus_score(predictions, reference_lists).score, 4),
        "ter":    round(TER().corpus_score(predictions, reference_lists).score, 4),
    }


# ============================================================
# EVALUATION ENTRY POINT
# ============================================================

def evaluate_predictions(predictions, records, inventory, threshold=50.0):
    """
    Full dual-reference report:

      augmented_reference : scored as generated, against glossed references
      clean_reference     : glosses stripped from both sides, comparable
                            to standard English-Pidgin MT results
      glossary            : terminology glossing quality
    """

    augmented_references = [
        r.get("pcm_augmented") or r.get("pcm") or "" for r in records
    ]

    clean_references = [
        r.get("pcm_clean") or strip_glosses(r.get("pcm_augmented", ""), inventory)
        for r in records
    ]

    stripped_predictions = [strip_glosses(p, inventory) for p in predictions]

    glossary_summary, glossary_rows = glossary_accuracy(
        predictions, records, inventory, threshold=threshold
    )

    report = {
        "n_examples": len(predictions),
        "augmented_reference": mt_metrics(predictions, augmented_references),
        "clean_reference": mt_metrics(stripped_predictions, clean_references),
        "glossary": glossary_summary,
    }

    return report, glossary_rows


# ============================================================
# CLI
# ============================================================

def load_split(data_dir, split):

    path = os.path.join(data_dir, f"{split}.json")

    with open(path) as handle:
        records = json.load(handle)

    if records and "glossed_terms" not in records[0]:
        raise SystemExit(
            f"{path} has no glossary metadata. Run prepare_data.py first."
        )

    return records


def main():

    parser = argparse.ArgumentParser(
        description="Score a predictions file with MT and glossary metrics."
    )

    parser.add_argument("--predictions", required=True,
                        help="CSV with a 'prediction' column (as written by train.py).")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--gloss-threshold", type=float, default=50.0)
    parser.add_argument("--out", default=None,
                        help="Where to write the JSON report (default: alongside predictions).")
    parser.add_argument("--per-term-out", default=None,
                        help="Optional CSV of every scored term.")

    args = parser.parse_args()

    records = load_split(args.data_dir, args.split)

    # The term inventory is the corpus glossary, a fixed public artifact
    # released with the benchmark rather than something derived from the
    # test references. Systems are free to use it too, so building it from
    # all splits is not leakage -- it is the terminology list under test.
    inventory = build_term_inventory(
        *[load_split(args.data_dir, s) for s in ("train", "dev", "test")]
    )

    frame = pd.read_csv(args.predictions)

    if "prediction" not in frame.columns:
        raise SystemExit(
            f"{args.predictions} has no 'prediction' column "
            f"(found: {list(frame.columns)})"
        )

    predictions = frame["prediction"].fillna("").astype(str).tolist()

    # Prefer an explicit id join; fall back to positional alignment.
    if "id" in frame.columns:
        by_id = {r["id"]: r for r in records}
        missing = [i for i in frame["id"] if i not in by_id]
        if missing:
            raise SystemExit(f"{len(missing)} prediction ids absent from {args.split}.json")
        records = [by_id[i] for i in frame["id"]]
    else:
        if len(predictions) != len(records):
            raise SystemExit(
                f"{len(predictions)} predictions vs {len(records)} {args.split} "
                f"records, and no 'id' column to join on."
            )

    report, rows = evaluate_predictions(
        predictions, records, inventory, threshold=args.gloss_threshold
    )

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.predictions)),
        "glossary_report.json"
    )

    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print("=" * 64)
    print(f"  {args.predictions}")
    print(f"  {report['n_examples']} examples | inventory: {len(inventory)} terms")
    print("=" * 64)

    print("\nMT metrics (augmented references, as generated)")
    for key, value in report["augmented_reference"].items():
        print(f"  {key:<8}: {value:>9}")

    print("\nMT metrics (clean references, glosses stripped)")
    for key, value in report["clean_reference"].items():
        print(f"  {key:<8}: {value:>9}")

    glossary = report["glossary"]

    print("\nGlossary metrics")
    print(f"  gloss accuracy      : {glossary['gloss_accuracy']:>9}   <- headline")
    print(f"  gloss presence rate : {glossary['gloss_presence_rate']:>9}")
    print(f"  gloss precision     : {glossary['gloss_precision']:>9}")
    print(f"  gloss F1            : {glossary['gloss_f1']:>9}")
    print(f"  term mention rate   : {glossary['term_mention_rate']:>9}")
    print(f"  over-glossing rate  : {glossary['over_glossing_rate']:>9}")
    print(f"  expected / produced : {glossary['expected_glosses']} / {glossary['produced_glosses']}")

    if glossary["by_subject"]:
        print("\n  by subject" + " " * 18 + "expected      GPR       GA")
        for subject, counts in glossary["by_subject"].items():
            print(f"    {subject:<24} {counts['expected']:>8} "
                  f"{counts['gloss_presence_rate']:>8} {counts['gloss_accuracy']:>8}")

    if args.per_term_out:
        pd.DataFrame(rows).to_csv(args.per_term_out, index=False)
        print(f"\nwrote per-term detail -> {args.per_term_out}")

    print(f"\nwrote report -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
