# ============================================================
# SPLIT REGENERATION
# ============================================================
# The original train/dev/test.json carry only {"en", "pcm"}, where "pcm"
# is actually the glossary-augmented target. That is enough to train on
# but not to score glossary accuracy, which needs to know WHICH terms
# were glossed in each row.
#
# This script rebuilds the same three splits -- the exact same rows in
# the exact same partition -- with the full record from the glossary CSV:
#
#     id, en, pcm (clean), pcm_augmented, glossed_terms, n_glosses, subject
#
# The existing partition is recovered by matching each JSON record back to
# its CSV row, so previously trained checkpoints stay comparable. Nothing
# is reshuffled.
#
# Usage:
#     python prepare_data.py
#     python prepare_data.py --dry-run
# ============================================================

import os
import json
import shutil
import argparse
import collections

import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SPLITS = ["train", "dev", "test"]


def norm(value):
    """CSV round-trips can leave NaN or stray whitespace; make keys stable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def parse_terms(raw):
    """'controversies; contexts' -> ['controversies', 'contexts']"""
    raw = norm(raw)
    if not raw:
        return []
    return [term.strip() for term in raw.split(";") if term.strip()]


def main():

    parser = argparse.ArgumentParser(
        description="Rebuild the splits with glossary metadata intact."
    )
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")

    args = parser.parse_args()

    data_dir = args.data_dir
    csv_path = args.csv or os.path.join(data_dir, "Eng-PidginEdu_glossary_augmented.csv")
    backup_dir = args.backup_dir or os.path.join(data_dir, "splits_backup")

    # ------------------------------------------------------------
    # Load the source of truth
    # ------------------------------------------------------------

    df = pd.read_csv(csv_path)

    print(f"glossary CSV : {len(df)} rows")

    # Index CSV rows by (en, pcm_augmented). A handful of rows share a
    # pcm_augmented string, so the English side is folded into the key and
    # any remaining collisions are consumed in order.
    index = collections.defaultdict(collections.deque)

    for position, row in enumerate(df.itertuples(index=False)):
        key = (norm(row.en), norm(row.pcm_augmented))
        index[key].append(position)

    # ------------------------------------------------------------
    # Recover the existing partition
    # ------------------------------------------------------------

    assignments = {}
    unmatched = collections.Counter()

    for split in SPLITS:

        split_path = os.path.join(data_dir, f"{split}.json")

        with open(split_path) as handle:
            records = json.load(handle)

        positions = []

        for record in records:

            key = (norm(record.get("en")), norm(record.get("pcm")))

            if index[key]:
                positions.append(index[key].popleft())
            else:
                unmatched[split] += 1

        assignments[split] = positions

        print(f"{split:<5} : {len(records)} records -> {len(positions)} matched")

    if unmatched:
        print("\nWARNING: unmatched records (left out of the rebuilt splits):")
        for split, count in unmatched.items():
            print(f"  {split}: {count}")

    # Every row should land in exactly one split.
    claimed = sum(len(v) for v in assignments.values())
    leftover = sum(len(v) for v in index.values())

    print(f"\nclaimed {claimed} / {len(df)} CSV rows  (unclaimed: {leftover})")

    overlap = set()
    seen = set()
    for split, positions in assignments.items():
        for position in positions:
            if position in seen:
                overlap.add(position)
            seen.add(position)

    if overlap:
        print(f"ERROR: {len(overlap)} rows assigned to more than one split")
        return 1

    # ------------------------------------------------------------
    # Build the enriched records
    # ------------------------------------------------------------

    def build(position):

        row = df.iloc[position]

        terms = parse_terms(row.get("glossed_terms"))

        return {
            "id":            f"engpidginedu-{position:05d}",
            "en":            norm(row.get("en")),
            # Kept as "pcm" so existing training runs are unaffected: this
            # is the augmented target the model is actually trained on.
            "pcm":           norm(row.get("pcm_augmented")),
            "pcm_clean":     norm(row.get("pcm")),
            "pcm_augmented": norm(row.get("pcm_augmented")),
            "glossed_terms": terms,
            "n_glosses":     len(terms),
            "subject":       norm(row.get("Subject")),
        }

    rebuilt = {
        split: [build(position) for position in positions]
        for split, positions in assignments.items()
    }

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    print()
    for split in SPLITS:
        records = rebuilt[split]
        glossed = sum(1 for r in records if r["n_glosses"] > 0)
        terms = sum(r["n_glosses"] for r in records)
        print(f"{split:<5} : {len(records):>6} rows | "
              f"{glossed:>6} glossed | {terms:>6} glossed terms")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    # ------------------------------------------------------------
    # Back up, then write
    # ------------------------------------------------------------

    os.makedirs(backup_dir, exist_ok=True)

    for split in SPLITS:

        source = os.path.join(data_dir, f"{split}.json")
        target = os.path.join(backup_dir, f"{split}.json")

        if os.path.exists(source) and not os.path.exists(target):
            shutil.copy2(source, target)

    print(f"\nbacked up original splits -> {backup_dir}")

    for split in SPLITS:

        out_path = os.path.join(data_dir, f"{split}.json")

        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(rebuilt[split], handle, ensure_ascii=False, indent=2)

        print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
