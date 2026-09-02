# ============================================================
# BENCHMARK LEADERBOARD
# ============================================================
# Collects every finished run in the project directory into one table.
#
# Reads   output_<model>/glossary_report.json
# Writes  benchmark_results.csv   (all columns, for analysis)
#         benchmark_results.md    (paper-ready table)
#
# Two MT columns are reported per model:
#
#   clean       glosses stripped from both hypothesis and reference.
#               Comparable to other English-Pidgin MT work -- this is the
#               number to put in a standard MT results table.
#
#   augmented   scored as generated against glossed references. Higher,
#               and only comparable within this benchmark.
#
# Usage:
#     python aggregate_results.py
#     python aggregate_results.py --sort gloss_f1
# ============================================================

import os
import json
import glob
import argparse

import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def collect(data_dir):
    """One row per finished run."""

    rows = []

    for report_path in sorted(glob.glob(os.path.join(data_dir, "output_*", "glossary_report.json"))):

        with open(report_path) as handle:
            report = json.load(handle)

        output_dir = os.path.dirname(report_path)

        clean = report.get("clean_reference", {})
        augmented = report.get("augmented_reference", {})
        glossary = report.get("glossary", {})

        # Training-time metrics, if the run got far enough to write them.
        metrics_path = os.path.join(output_dir, "metrics.json")
        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as handle:
                metrics = json.load(handle)

        # AfriCOMET, if africomet_metrics.py has been run for this model
        # (separate step -- not part of train.py's own end-of-run writes,
        # since it needs a second model loaded and is meant to be re-runnable
        # without retraining).
        africomet_path = os.path.join(output_dir, "africomet_report.json")
        africomet = {}
        if os.path.exists(africomet_path):
            with open(africomet_path) as handle:
                africomet = json.load(handle)

        dir_name = os.path.basename(output_dir)

        # The model name alone is not enough to tell rows apart -- the
        # same model appears once per condition (full fine-tune, LoRA,
        # zero-shot, and occasionally a validation-split re-score), each
        # with its own row here. Derived from the directory naming
        # convention rather than a field in the report itself, since
        # zero-shot/re-score runs don't consistently write one.
        if dir_name.startswith("output_lora_"):
            condition = "LoRA"
        elif dir_name.startswith("output_zeroshot_"):
            condition = "Zero-shot"
        elif dir_name.endswith("_dev"):
            condition = "Full fine-tune (dev-split re-score)"
        else:
            condition = "Full fine-tune"

        rows.append({
            "model":          report.get("model", dir_name.replace("output_", "")),
            "condition":      condition,
            "checkpoint":     report.get("model_name", ""),
            "target_column":  report.get("target_column", "") or "pcm_augmented",
            "n_examples":     report.get("n_examples"),

            # Standard-comparable MT scores.
            "bleu":           clean.get("bleu"),
            "chrf":           clean.get("chrf"),
            "chrf++":         clean.get("chrf++"),
            "ter":            clean.get("ter"),
            "africomet":      africomet.get("africomet_stl"),

            # Within-benchmark scores against glossed references.
            "bleu_aug":       augmented.get("bleu"),
            "chrf++_aug":     augmented.get("chrf++"),
            "ter_aug":        augmented.get("ter"),

            # Terminology.
            "gloss_accuracy": glossary.get("gloss_accuracy"),
            "gloss_presence": glossary.get("gloss_presence_rate"),
            "gloss_precision": glossary.get("gloss_precision"),
            "gloss_f1":       glossary.get("gloss_f1"),
            "over_glossing":  glossary.get("over_glossing_rate"),

            "test_loss":      metrics.get("test_loss"),
            "output_dir":     os.path.basename(output_dir),
        })

    return pd.DataFrame(rows)


def to_markdown(frame):
    """Paper-ready table: the columns a reader actually needs."""

    columns = [
        ("model", "Model"),
        ("condition", "Condition"),
        ("checkpoint", "Checkpoint"),
        ("bleu", "BLEU"),
        ("chrf++", "chrF++"),
        ("ter", "TER"),
        ("africomet", "AfriCOMET"),
        ("gloss_accuracy", "GlossAcc"),
        ("gloss_f1", "GlossF1"),
    ]

    present = [(key, label) for key, label in columns if key in frame.columns]

    lines = [
        "| " + " | ".join(label for _, label in present) + " |",
        "|" + "|".join("---" for _ in present) + "|",
    ]

    for _, row in frame.iterrows():
        cells = []
        for key, _ in present:
            value = row[key]
            cells.append("N/A" if pd.isna(value) else
                         (f"{value:.2f}" if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main():

    parser = argparse.ArgumentParser(description="Aggregate benchmark runs.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--sort", default="gloss_f1",
                        help="Column to rank by (default: gloss_f1).")

    args = parser.parse_args()

    frame = collect(args.data_dir)

    if frame.empty:
        print("No finished runs found. Each run writes "
              "output_<model>/glossary_report.json when it completes.")
        return 0

    if args.sort in frame.columns:
        frame = frame.sort_values(args.sort, ascending=False, na_position="last")

    csv_path = os.path.join(args.data_dir, "benchmark_results.csv")
    md_path = os.path.join(args.data_dir, "benchmark_results.md")

    frame.to_csv(csv_path, index=False)

    markdown = to_markdown(frame)

    with open(md_path, "w") as handle:
        handle.write("# Eng-PidginEdu benchmark\n\n")
        handle.write("MT scores are against **clean references** "
                     "(glosses stripped from both sides), so they are "
                     "comparable to standard English-Pidgin MT results.\n\n")
        handle.write("The same model name can appear more than once -- "
                     "the **Condition** column is what distinguishes a "
                     "full fine-tune, a LoRA run, a zero-shot evaluation, "
                     "and (rare) a re-score against the dev split for the "
                     "same model. All runs train against and are scored "
                     "against the `pcm_augmented` column of the dataset; "
                     "`benchmark_results.csv` also has a raw `output_dir` "
                     "column if you need the exact directory a row came "
                     "from.\n\n")
        handle.write(markdown + "\n")

    print(f"{len(frame)} run(s)\n")
    print(markdown)
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
