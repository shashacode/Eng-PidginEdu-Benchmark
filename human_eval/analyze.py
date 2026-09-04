# ============================================================
# PidginEdu human evaluation: analysis
# ============================================================
# Run after collecting one or more rater JSON files downloaded from
# pidginedu_rating_study.html.
#
# Usage:
#     python analyze.py pidginedu_ratings_*.json
#
# Unblinds each rater's A/B choice against unblind_key.json (never
# shipped to raters), reports per-model win rate, ties, gloss-judgment
# breakdown, and (with 2+ raters) pairwise inter-rater agreement.
# ============================================================

import sys
import json
import glob
from collections import Counter, defaultdict
from itertools import combinations

KEY_PATH = "unblind_key.json"


def load_key():
    with open(KEY_PATH) as f:
        rows = json.load(f)
    return {r["id"]: r for r in rows}


def unblind_choice(choice, key_row):
    """Map a rater's 'a'/'b'/'tie' choice to the real model name."""
    if choice == "tie":
        return "tie"
    if choice == "a":
        return key_row["a_model"]
    if choice == "b":
        return key_row["b_model"]
    return None


def unblind_gloss(gloss, key_row):
    """Map a gloss judgment referencing a/b into model terms where it
    names one side specifically; symmetric judgments pass through."""
    if gloss in (None, "both_good", "neither", "na"):
        return gloss
    if gloss == "a_only":
        return key_row["a_model"] + "_only"
    if gloss == "b_only":
        return key_row["b_model"] + "_only"
    return gloss


def load_rater_file(path, key):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    unblinded = {}
    for row in payload["ratings"]:
        row_id = row["row_id"]
        key_row = key.get(row_id)
        if key_row is None:
            continue
        unblinded[row_id] = {
            "choice": unblind_choice(row.get("choice"), key_row),
            "gloss": unblind_gloss(row.get("gloss_judgment"), key_row),
            "note": row.get("note", ""),
        }

    return {
        "rater_name": payload.get("rater_name", path),
        "answered": payload.get("answered_items"),
        "total": payload.get("total_items"),
        "unblinded": unblinded,
    }


def preference_summary(raters):
    counts = Counter()
    for r in raters:
        for row in r["unblinded"].values():
            if row["choice"]:
                counts[row["choice"]] += 1
    total = sum(counts.values())
    print(f"\n=== Preference, pooled across {len(raters)} rater(s), {total} judgments ===")
    for model, n in counts.most_common():
        print(f"  {model:<12} {n:>4}  ({100 * n / total:.1f}%)")


def gloss_summary(raters):
    counts = Counter()
    for r in raters:
        for row in r["unblinded"].values():
            if row["gloss"] and row["gloss"] not in ("na",):
                counts[row["gloss"]] += 1
    total = sum(counts.values())
    if total == 0:
        print("\n=== Gloss judgments: none recorded ===")
        return
    print(f"\n=== Gloss judgment, pooled, {total} judgments ===")
    for label, n in counts.most_common():
        print(f"  {label:<20} {n:>4}  ({100 * n / total:.1f}%)")


def cohen_kappa(pairs):
    """pairs: list of (rating_a, rating_b) tuples for the same items
    from two raters, categorical labels. Unweighted Cohen's kappa."""
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted(set(x for pair in pairs for x in pair))
    po = sum(1 for a, b in pairs if a == b) / n

    marg_a = Counter(a for a, _ in pairs)
    marg_b = Counter(b for _, b in pairs)
    pe = sum((marg_a[l] / n) * (marg_b[l] / n) for l in labels)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def agreement_summary(raters):
    if len(raters) < 2:
        print("\n=== Inter-rater agreement: need 2+ raters, only have "
              f"{len(raters)} ===")
        return

    print(f"\n=== Pairwise inter-rater agreement (Cohen's kappa, preference) ===")
    for r1, r2 in combinations(raters, 2):
        shared = set(r1["unblinded"]) & set(r2["unblinded"])
        pairs = []
        for row_id in shared:
            c1 = r1["unblinded"][row_id]["choice"]
            c2 = r2["unblinded"][row_id]["choice"]
            if c1 and c2:
                pairs.append((c1, c2))
        kappa = cohen_kappa(pairs)
        label = f"{r1['rater_name']} vs {r2['rater_name']}"
        if kappa is None:
            print(f"  {label}: no overlapping rated items")
        else:
            print(f"  {label}: kappa={kappa:.3f}  (n={len(pairs)} shared items)")


def flagged_notes(raters):
    notes = []
    for r in raters:
        for row_id, row in r["unblinded"].items():
            if row["note"].strip():
                notes.append((r["rater_name"], row_id, row["note"]))
    if not notes:
        return
    print(f"\n=== Flagged notes ({len(notes)}) ===")
    for rater, row_id, note in notes:
        print(f"  [{rater} / {row_id}] {note}")


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = sorted(glob.glob("pidginedu_ratings_*.json"))
    if not paths:
        print("No rater files found. Pass paths, or place "
              "pidginedu_ratings_*.json files in this directory.")
        return 1

    key = load_key()
    raters = [load_rater_file(p, key) for p in paths]

    print(f"Loaded {len(raters)} rater file(s):")
    for r in raters:
        print(f"  {r['rater_name']}: {r['answered']}/{r['total']} answered ({paths[raters.index(r)]})")

    preference_summary(raters)
    gloss_summary(raters)
    agreement_summary(raters)
    flagged_notes(raters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
