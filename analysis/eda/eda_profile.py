"""Memory-conscious profile of every source file (one file in memory at a time)."""
import gc
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

BASE = r"C:\Users\ANEXUS\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
COLS = ["entity_id", "business_name", "business_address", "country"]


def read(path, cols):
    return pv.read_csv(
        path,
        parse_options=pv.ParseOptions(delimiter="\t"),
        convert_options=pv.ConvertOptions(
            column_types={c: pa.string() for c in cols},
            strings_can_be_null=False,
            null_values=[],
        ),
        read_options=pv.ReadOptions(block_size=1 << 26),
    )


def frac(mask, n):
    k = pc.sum(mask).as_py() or 0
    return f"{k:>9,} ({100 * k / n:5.2f}%)"


def top(arr, k=25):
    vc = pc.value_counts(arr)
    vals, cnts = vc.field("values"), vc.field("counts")
    idx = pc.array_sort_indices(cnts, order="descending")[:k]
    return [(vals[i].as_py(), cnts[i].as_py()) for i in idx.to_pylist()]


def profile(name):
    t = time.time()
    tb = read(f"{BASE}\\{name}", COLS)
    n = tb.num_rows
    eid, nm, ad, co = (tb.column(c).combine_chunks() for c in COLS)
    print(f"\n{'=' * 90}\n{name}: {n:,} rows  (read {time.time() - t:.1f}s)")
    print(f"  unique entity_id: {pc.count_distinct(eid).as_py():,}")
    pref = pc.utf8_slice_codeunits(eid, 0, 3)
    print(f"  id prefixes: {top(pref, 5)}")
    idlen = pc.utf8_length(eid)
    print(f"  id length min/max: {pc.min(idlen).as_py()}/{pc.max(idlen).as_py()}")
    print(f"  country: {top(co, 10)}")

    nm_s, ad_s = pc.utf8_trim_whitespace(nm), pc.utf8_trim_whitespace(ad)
    print(f"  empty name         : {frac(pc.equal(nm_s, ''), n)}")
    print(f"  empty address      : {frac(pc.equal(ad_s, ''), n)}")
    for tok in ["null", "None", "nan", "N/A", "NA", "-"]:
        m = pc.equal(pc.utf8_lower(ad_s), tok.lower())
        if pc.sum(m).as_py():
            print(f"  address == {tok!r:6}  : {frac(m, n)}")
    print(f"  addr has 'null' tok: {frac(pc.match_substring_regex(ad, r'(?i)(^|[ ,])null([ ,]|$)'), n)}")
    print(f"  name has 'null' tok: {frac(pc.match_substring_regex(nm, r'(?i)(^|[ ,])null([ ,]|$)'), n)}")

    nl, al = pc.utf8_length(nm), pc.utf8_length(ad)
    q = [0.01, 0.25, 0.5, 0.75, 0.99]
    print(f"  name len q{q}: {pc.quantile(nl, q=q).to_pylist()}")
    print(f"  addr len q{q}: {pc.quantile(al, q=q).to_pylist()}")

    for label, rx in [
        ("Devanagari", r"\p{Devanagari}"),
        ("other Indic", r"[\p{Bengali}\p{Gurmukhi}\p{Gujarati}\p{Oriya}\p{Tamil}\p{Telugu}\p{Kannada}\p{Malayalam}]"),
        ("Latin accents", r"[À-ÖØ-öø-ÿŒœ]"),
        ("all-UPPER name", r"^[^a-z]*[A-Z][^a-z]*$"),
        ("digit in name", r"\d"),
        ("url/www/.com", r"(?i)(www\.|\.com|\.in\b|\.fr\b|https?:)"),
        ("junk <<,>>,--,|,*,#", r"(<<|>>|--|\||\*|#)"),
        ("brackets ()[]{}", r"[\(\)\[\]\{\}]"),
        ("& in name", r"&"),
        ("DBA/aka/t-a", r"(?i)\b(dba|d/b/a|aka|t/a|trading as)\b"),
    ]:
        print(f"  name {label:18}: {frac(pc.match_substring_regex(nm, rx), n)}")
    for label, rx in [
        ("Devanagari", r"\p{Devanagari}"),
        ("other Indic", r"[\p{Bengali}\p{Gurmukhi}\p{Gujarati}\p{Oriya}\p{Tamil}\p{Telugu}\p{Kannada}\p{Malayalam}]"),
        ("6-digit (PIN)", r"(^|\D)\d{6}(\D|$)"),
        ("5-digit (ZIP/CP)", r"(^|\D)\d{5}(\D|$)"),
        ("ZIP+4", r"\d{5}-\d{4}"),
        ("'near'/'opp'", r"(?i)\b(near|nr\.?|opp\.?|opposite|behind|beside)\b"),
        ("PO Box", r"(?i)\bP\.?\s?O\.?\s?Box\b"),
        ("Unit/Apt/Suite", r"(?i)\b(unit|apt|suite|ste)\b"),
        ("all-UPPER addr", r"^[^a-z]*[A-Z][^a-z]*$"),
        ("starts w/ state/abbr", r"^[A-Z]{2},"),
    ]:
        print(f"  addr {label:18}: {frac(pc.match_substring_regex(ad, rx), n)}")

    # name-level duplication within the source (how "common" names are)
    low = pc.utf8_lower(nm_s)
    vc = pc.value_counts(low).field("counts")
    print(f"  distinct lower(name): {len(vc):,}  | names used >1x: {pc.sum(pc.greater(vc, 1)).as_py():,}"
          f"  | max reuse: {pc.max(vc).as_py():,}")
    print(f"  top names: {top(low, 12)}")

    # token vocabulary of names
    toks = pc.list_flatten(pc.split_pattern_regex(low, r"[^\p{L}\p{N}&+]+"))
    toks = pc.filter(toks, pc.not_equal(toks, ""))
    print(f"  top 60 name tokens: {top(toks, 60)}")
    atoks = pc.list_flatten(pc.split_pattern_regex(pc.utf8_lower(ad), r"[^\p{L}\p{N}]+"))
    atoks = pc.filter(atoks, pc.not_equal(atoks, ""))
    print(f"  top 60 addr tokens: {top(atoks, 60)}")
    ids = eid
    del tb, nm, ad, co, toks, atoks, low
    gc.collect()
    return ids


if __name__ == "__main__":
    files = sys.argv[1:] or [
        "train\\train_source1.tsv", "train\\train_source2.tsv", "train\\train_source3.tsv",
        "test\\test_source1.tsv", "test\\test_source2.tsv", "test\\test_source3.tsv",
    ]
    for f in files:
        profile(f)
        gc.collect()
