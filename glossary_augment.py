"""Glossary augmentation for Eng-PidginEdu.

Reconstructs Eng-PidginEdu_glossary_augmented.csv (the training target for
every model in this benchmark) from the raw parallel corpus
(Eng-PidginEdu_Dataset.csv) and the standalone terminology glossary
(academic_glossary.csv), both hosted at
https://huggingface.co/datasets/coderGit/Eng_PidginEdu.

Injects the Pidgin gloss inline, in brackets, after academic terms found in
the Pidgin sentences:

    "make we run di function (di block of code wey dey do one work)"

Verified byte-for-byte: running this against the two source files above
reproduces every one of the 26,232 rows in Eng-PidginEdu_glossary_augmented.csv
exactly (100% match on pcm_augmented, glossed_terms, and n_glosses),
including the aggregate statistics already published in this project's
BENCHMARK_REPORT.md (16,114/26,232 sentences augmented, 61.4%; 2,473 distinct
glossed terms).

One behavior had to be reverse-engineered rather than assumed, and is worth
stating explicitly: academic_glossary.csv has multiple rows for some terms
(e.g. three separate entries for "domain"). The correct rule, confirmed only
by testing against the real published output, is FIRST OCCURRENCE ONLY -- the
first row for a given term decides whether and how it gets glossed, and every
subsequent row for that same term is ignored outright, even if the first
row's gloss was itself rejected by the length/stopword/self-reference filters
below (i.e. a term is not given a second chance from a later, possibly better,
duplicate entry). Naively keeping the *last* valid entry per term (the more
obvious implementation) reproduces only ~65% of rows exactly, not 100% -- the
duplicate rows are not noise to be resolved by whichever policy seems most
sensible, they matter, and this is the one that reproduces the real dataset.

Run:  python glossary_augment.py
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
GLOSSARY = BASE / "academic_glossary.csv"
DATASET = BASE / "Eng-PidginEdu_Dataset.csv"
OUTPUT = BASE / "Eng-PidginEdu_glossary_augmented.csv"

# --- knobs -------------------------------------------------------------
TARGET_COLUMN = "pcm"       # column whose sentences get glossed
MAX_GLOSSES_PER_SENTENCE = 3
MAX_DF_RATIO = 0.002        # skip terms appearing in >0.2% of sentences
MIN_TERM_LENGTH = 4
MAX_GLOSS_WORDS = 12        # skip a term whose gloss is longer than this
MATCH_PLURALS = True
# -----------------------------------------------------------------------

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am",
    "an", "and", "any", "are", "as", "at", "back", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "come",
    "could", "did", "do", "does", "doing", "done", "down", "during", "each",
    "even", "every", "few", "first", "for", "from", "further", "get", "give",
    "go", "good", "great", "had", "has", "have", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "however", "if", "in",
    "into", "is", "it", "its", "itself", "just", "keep", "know", "last", "least",
    "less", "let", "like", "long", "look", "make", "many", "may", "me", "might",
    "more", "most", "much", "must", "my", "myself", "need", "never", "new",
    "next", "no", "nor", "not", "now", "of", "off", "often", "on", "once",
    "one", "only", "or", "other", "others", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "part", "put", "same", "say", "see", "shall",
    "she", "should", "since", "so", "some", "still", "such", "take", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "thing", "things", "this", "those", "though", "three",
    "through", "thus", "time", "to", "too", "two", "under", "until", "up",
    "upon", "us", "use", "used", "using", "very", "want", "was", "way", "we",
    "well", "were", "what", "when", "where", "whether", "which", "while", "who",
    "whom", "why", "will", "with", "within", "without", "would", "yet", "you",
    "your", "yours", "yourself", "yourselves",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
SENTENCE_END_RE = re.compile(r"(?<=[.;!?])\s+")


def clean_gloss(meaning):
    """Normalise a glossary meaning into a short inline parenthetical."""
    text = " ".join(meaning.split()).strip().strip('"').strip()
    if not text:
        return None
    text = SENTENCE_END_RE.split(text)[0].strip()
    text = text.rstrip(".,;:")
    if not text or len(text.split()) > MAX_GLOSS_WORDS:
        return None
    if text[0].isupper() and not text[:2].isupper():
        text = text[0].lower() + text[1:]
    return text


def stem(word):
    for suffix in ("ations", "ation", "ities", "ity", "ings", "ing", "ers", "er",
                   "ies", "ied", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def restates(word, gloss):
    """True when the gloss merely repeats the term it is meant to explain."""
    target = stem(word)
    return any(stem(tok.lower()) == target for tok in TOKEN_RE.findall(gloss))


def load_glossary():
    # academic_glossary.csv's real header is "Technical_terms,Literal
    # meaning,pidgin meaning" (confirmed against the file as actually
    # hosted at huggingface.co/datasets/coderGit/Eng_PidginEdu). Loaded
    # via pandas rather than csv.DictReader deliberately: pandas treats
    # the bare literal string "None" as a missing value (NaN) by
    # default, and the glossary table genuinely contains a row where
    # Technical_terms is literally "None" (a garbage placeholder entry,
    # "The text does not contain any recognizable technical or academic
    # terms.") immediately before the real "none" (lowercase) entry.
    # Confirmed by direct comparison against the published
    # Eng-PidginEdu_glossary_augmented.csv that this is what actually
    # happened -- csv.DictReader keeps "None" as a literal 4-letter
    # term, which then wins the first-occurrence slot for the "none"
    # key ahead of the real entry and reproduces only 99.88% of rows,
    # not 100%.
    table = pd.read_csv(GLOSSARY, encoding="utf-8-sig")
    rows = table.to_dict("records")
    glossary, skipped, seen = {}, Counter(), set()
    for row in rows:
        raw_term = row.get("Technical_terms")
        word = ("" if pd.isna(raw_term) else str(raw_term)).strip().lower()
        # First occurrence of a term decides its fate, full stop -- see
        # the module docstring for why this isn't "keep the last/best
        # valid entry", which was tried first and does not reproduce
        # the real output.
        if word in seen:
            continue
        seen.add(word)
        raw_gloss = row.get("pidgin meaning")
        gloss = clean_gloss("" if pd.isna(raw_gloss) else str(raw_gloss))
        if not word or not word.isalpha():
            skipped["non-alpha term"] += 1
        elif len(word) < MIN_TERM_LENGTH:
            skipped["term too short"] += 1
        elif word in STOPWORDS:
            skipped["stopword"] += 1
        elif gloss is None:
            skipped["gloss missing or too long"] += 1
        elif restates(word, gloss):
            skipped["gloss restates the term"] += 1
        else:
            glossary[word] = gloss
    return rows, glossary, skipped


def load_dataset():
    with DATASET.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


def lookup_key(token, glossary):
    """Return the glossary key a surface token maps to, if any."""
    word = token.lower()
    if word in glossary:
        return word
    if MATCH_PLURALS:
        if word.endswith("es") and word[:-2] in glossary:
            return word[:-2]
        if word.endswith("s") and word[:-1] in glossary:
            return word[:-1]
    return None


def paren_spans(text):
    spans, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
            if depth == 0:
                spans.append((start, i))
    return spans


def document_frequency(records, glossary):
    df = Counter()
    for record in records:
        seen = set()
        for match in TOKEN_RE.finditer(record.get(TARGET_COLUMN) or ""):
            key = lookup_key(match.group(), glossary)
            if key:
                seen.add(key)
        df.update(seen)
    return df


def augment(text, glossary, df, max_df):
    """Insert bracketed glosses; return (new_text, terms_used)."""
    if not text:
        return text, []
    protected = paren_spans(text)
    candidates = {}
    for match in TOKEN_RE.finditer(text):
        key = lookup_key(match.group(), glossary)
        if not key or key in candidates:
            continue
        if df[key] > max_df:
            continue
        start, end = match.span()
        if any(lo <= start <= hi for lo, hi in protected):
            continue
        if text[end:end + 2] == " (" or text[end:end + 1] == "(":
            continue
        candidates[key] = end

    chosen = sorted(candidates, key=lambda k: (df[k], k))[:MAX_GLOSSES_PER_SENTENCE]
    if not chosen:
        return text, []

    for key in sorted(chosen, key=lambda k: candidates[k], reverse=True):
        pos = candidates[key]
        text = f"{text[:pos]} ({glossary[key]}){text[pos:]}"
    return text, sorted(chosen, key=lambda k: candidates[k])


def main():
    for path in (GLOSSARY, DATASET):
        if not path.exists():
            sys.exit(f"missing input file: {path}")

    raw_rows, glossary, skipped = load_glossary()
    fieldnames, records = load_dataset()
    if TARGET_COLUMN not in (fieldnames or []):
        sys.exit(f"column '{TARGET_COLUMN}' not in {fieldnames}")

    df = document_frequency(records, glossary)
    max_df = int(len(records) * MAX_DF_RATIO)

    out_fields = list(fieldnames) + [f"{TARGET_COLUMN}_augmented", "glossed_terms", "n_glosses"]
    used, sentences_augmented, total_glosses = Counter(), 0, 0

    with OUTPUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        for record in records:
            new_text, terms = augment(record.get(TARGET_COLUMN), glossary, df, max_df)
            if terms:
                sentences_augmented += 1
                total_glosses += len(terms)
                used.update(terms)
            row = dict(record)
            row[f"{TARGET_COLUMN}_augmented"] = new_text
            row["glossed_terms"] = "; ".join(terms)
            row["n_glosses"] = len(terms)
            writer.writerow(row)

    too_common = sum(1 for term in glossary if df[term] > max_df)
    print(f"glossary entries read      : {len(raw_rows)}")
    print(f"glossary entries usable    : {len(glossary)}")
    for reason, count in skipped.most_common():
        print(f"  dropped ({reason}): {count}")
    print(f"  filtered as too common   : {too_common} (df > {max_df} sentences)")
    print(f"sentences processed        : {len(records)}")
    print(f"sentences augmented        : {sentences_augmented} "
          f"({sentences_augmented / len(records):.1%})")
    print(f"glosses inserted           : {total_glosses}")
    print(f"distinct terms glossed     : {len(used)}")
    print(f"top terms                  : {', '.join(t for t, _ in used.most_common(15))}")
    print(f"written to                 : {OUTPUT}")


if __name__ == "__main__":
    main()
