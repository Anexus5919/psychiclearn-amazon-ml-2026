"""Sample clusters + similarity signals of matched pairs vs random same-country pairs."""
import gc
import random
import re
import unicodedata
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
from rapidfuzz import fuzz

BASE = r"C:\Users\ANEXUS\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
COLS = ["entity_id", "business_name", "business_address", "country"]
random.seed(0)
np.random.seed(0)


def read(path, cols):
    return pv.read_csv(
        path,
        parse_options=pv.ParseOptions(delimiter="\t"),
        convert_options=pv.ConvertOptions(
            column_types={c: pa.string() for c in cols},
            strings_can_be_null=False, null_values=[], include_columns=cols),
        read_options=pv.ReadOptions(block_size=1 << 26),
    )


def fetch(path, ids):
    """Return {id: (name, addr, country)} for the requested ids only."""
    tb = read(path, COLS)
    mask = pc.is_in(tb.column("entity_id"), value_set=pa.array(list(ids)))
    sub = tb.filter(mask).to_pylist()
    del tb
    gc.collect()
    return {r["entity_id"]: (r["business_name"], r["business_address"], r["country"]) for r in sub}


LEGAL = set("""pvt private ltd limited llc inc incorporated corp corporation co company llp lp pc pllc plc
the and of sa sas sarl sasu eurl sci snc sca pty opc""".split())
DEVA = re.compile(r"[\u0900-\u097F]")
INDIC = re.compile(r"[\u0980-\u0DFF]")


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("&", " and ")
    return re.sub(r"[^\w]+", " ", s).strip()


def core_tokens(s):
    return [t for t in norm(s).split() if t not in LEGAL]


def nums(s):
    return set(re.findall(r"\d+", s))


gt = read(f"{BASE}\\train\\train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"]).to_pylist()
sample = random.sample(gt, 30000)
del gt
gc.collect()
need1 = {r["source1_entity_id"] for r in sample}
need2, need3 = set(), set()
for r in sample:
    for m in (r["matched_entity_ids"] or "").split(","):
        if m.startswith("S2-"):
            need2.add(m)
        elif m.startswith("S3-"):
            need3.add(m)
R = {}
R.update(fetch(f"{BASE}\\train\\train_source1.tsv", need1))
S1_ALL_IDS = list(need1)
R.update(fetch(f"{BASE}\\train\\train_source2.tsv", need2))
R.update(fetch(f"{BASE}\\train\\train_source3.tsv", need3))
print(f"fetched {len(R):,} records for {len(sample):,} sampled clusters\n")

# ---------- print sample clusters ------------------------------------------------------
shown = Counter()
print("=" * 100 + "\nSAMPLE CLUSTERS\n" + "=" * 100)
for r in sample:
    s1 = R[r["source1_entity_id"]]
    ms = [m for m in (r["matched_entity_ids"] or "").split(",") if m]
    key = (s1[2], "single" if not ms else "multi")
    if shown[key] >= (6 if key[1] == "single" else 16):
        continue
    shown[key] += 1
    print(f"\n[{s1[2]}] {r['source1_entity_id']}: {s1[0]!r} | {s1[1]!r}")
    if not ms:
        print("    (singleton)")
    for m in ms:
        x = R[m]
        print(f"    {m[:2]} {m}: {x[0]!r} | {x[1]!r} [{x[2]}]")

# ---------- pair signals -----------------------------------------------------------------
pos_pairs = []
for r in sample:
    for m in (r["matched_entity_ids"] or "").split(","):
        if m:
            pos_pairs.append((r["source1_entity_id"], m))
by_country_src = {}
for m in list(need2) + list(need3):
    by_country_src.setdefault((R[m][2], m[:2]), []).append(m)
neg_pairs = []
for a, m in pos_pairs:
    pool = by_country_src[(R[a][2], m[:2])]
    neg_pairs.append((a, random.choice(pool)))


def signals(pairs):
    rows = []
    for a, b in pairs:
        (n1, a1, c1), (n2, a2, c2) = R[a], R[b]
        t1, t2 = set(core_tokens(n1)), set(core_tokens(n2))
        rows.append(dict(
            src=b[:2], country=c1,
            exact_lower=n1.lower() == n2.lower(),
            norm_eq=" ".join(sorted(t1)) == " ".join(sorted(t2)),
            share_core_tok=bool(t1 & t2),
            name_jacc=len(t1 & t2) / max(1, len(t1 | t2)),
            name_tsr=fuzz.token_set_ratio(norm(n1), norm(n2)),
            name_tsort=fuzz.token_sort_ratio(norm(n1), norm(n2)),
            deva=bool(DEVA.search(n2)), indic=bool(INDIC.search(n2)),
            addr_tsr=fuzz.token_set_ratio(norm(a1), norm(a2)),
            addr_empty=not a2.strip(),
            share_num=bool(nums(a1) & nums(a2)),
            num_both=bool(nums(a1)) and bool(nums(a2)),
        ))
    return rows


def summarize(rows, label):
    print(f"\n--- {label}: n={len(rows):,}")
    for grp in sorted({(r["country"], r["src"]) for r in rows}):
        g = [r for r in rows if (r["country"], r["src"]) == grp]
        f = lambda k: np.mean([r[k] for r in g])
        q = lambda k: np.percentile([r[k] for r in g], [10, 25, 50, 75, 90]).round(0).tolist()
        nb = [r for r in g if r["num_both"]]
        print(f"  {grp[0]:6} {grp[1]}  n={len(g):>6,} | exact_lower={f('exact_lower'):.3f} norm_eq={f('norm_eq'):.3f} "
              f"share_core_tok={f('share_core_tok'):.3f} name_devanagari={f('deva'):.3f} other_indic={f('indic'):.3f} "
              f"addr_empty={f('addr_empty'):.3f}")
        print(f"         name token_set_ratio p10/25/50/75/90 = {q('name_tsr')}   token_sort = {q('name_tsort')}")
        print(f"         addr token_set_ratio p10/25/50/75/90 = {q('addr_tsr')}   "
              f"share_number(|both have nums)={np.mean([r['share_num'] for r in nb]) if nb else float('nan'):.3f}")


pos = signals(pos_pairs)
neg = signals(neg_pairs)
summarize(pos, "MATCHED pairs")
summarize(neg, "RANDOM same-country/same-source pairs")

# how many matched pairs have NO shared core name token and no Devanagari -> hard for token blocking
hard = [(a, b) for (a, b), r in zip(pos_pairs, pos) if not r["share_core_tok"] and not r["deva"] and not r["indic"]]
print(f"\nmatched pairs with no shared core name token (Latin script both sides): {len(hard):,} / {len(pos):,}")
for a, b in random.sample(hard, min(40, len(hard))):
    print(f"  {R[a][0]!r:45} <-> {R[b][0]!r:45} | {R[a][1][:50]!r} <-> {R[b][1][:50]!r}")
