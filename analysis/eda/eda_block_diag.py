"""Diagnose low name-only recall (US, S2): is the true match out-scored/tied by other records that
share the same normalised core name, or is retrieval buggy? Brute-force cosines for 300 queries."""
import re
import unicodedata
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = r"C:\Users\ANEXUS\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
COLS = ["entity_id", "business_name", "business_address", "country"]
rng = np.random.default_rng(1)
DROP = set("""pvt private ltd limited llc inc incorporated corp corporation co company llp lp pc pllc plc
public the and of mr mrs ms m s shri sri smt""".split())
DBA = re.compile(r"\b(?:trading as|t/a|a/k/a|aka|f/k/a|fka|formerly|d/b/a|dba)\b", re.I)


def read(path, cols):
    return pv.read_csv(path, parse_options=pv.ParseOptions(delimiter="\t"),
                       convert_options=pv.ConvertOptions(column_types={c: pa.string() for c in cols},
                                                         strings_can_be_null=False, null_values=[],
                                                         include_columns=cols))


def norm_name(s):
    parts = DBA.split(s)
    if len(parts) > 1:
        s = parts[-1]
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(ch))
    s = re.sub(r"\.(com|in|net|org|co|fr)\b", " ", s).replace("&", " ")
    return " ".join(t for t in re.split(r"[^0-9a-z\u0900-\u0dff]+", s) if t and t not in DROP)


gt = read(f"{BASE}\\train\\train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
gtd = dict(zip(gt.column(0).to_pylist(), gt.column(1).to_pylist()))
s1 = read(f"{BASE}\\train\\train_source1.tsv", COLS)
s1 = s1.filter(pc.equal(s1.column("country"), "US"))
q = s1.take(pa.array(rng.choice(s1.num_rows, 300, replace=False))).to_pylist()
pool = read(f"{BASE}\\train\\train_source2.tsv", COLS)
pool = pool.filter(pc.equal(pool.column("country"), "US"))
pids = pool.column("entity_id").to_pylist()
praw = pool.column("business_name").to_pylist()
pn = [norm_name(x) for x in praw]
pos = {e: i for i, e in enumerate(pids)}
core_count = Counter(pn)

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
P = vec.fit_transform(pn).tocsr()
qn = [norm_name(r["business_name"]) for r in q]
Q = vec.transform(qn).tocsr()
PT = P.T.tocsr()

K = 20
miss = hit = 0
ties_or_outscored = Counter()
same_core_pool = []
examples = []
for r, rec in enumerate(q):
    true = [m for m in (gtd[rec["entity_id"]] or "").split(",") if m.startswith("S2-")]
    if not true:
        continue
    col = (Q[r] @ PT).toarray().ravel()  # exact brute-force cosines vs the whole pool
    kth = np.partition(col, -K)[-K]
    same_core_pool.append(core_count[qn[r]])
    for m in true:
        c = col[pos[m]]
        n_better = int((col > c).sum())
        if n_better < K:
            hit += 1
        else:
            miss += 1
            ties_or_outscored["true cos == 1.0 but >=K records also at 1.0" if c > 0.9999 else
                              "true cos < 1 and >=K records score higher"] += 1
            if len(examples) < 12:
                top = np.argsort(-col)[:3]
                examples.append((rec["business_name"], praw[pos[m]], round(float(c), 3), round(float(kth), 3),
                                 [(praw[i], round(float(col[i]), 3)) for i in top]))
print(f"exact brute-force name-only recall@{K}: {hit / (hit + miss):.4f}  (hits {hit}, misses {miss})")
print("miss reasons:", dict(ties_or_outscored))
print(f"#pool records sharing the query's exact core name: median={np.median(same_core_pool):.0f} "
      f"p75={np.percentile(same_core_pool, 75):.0f} p90={np.percentile(same_core_pool, 90):.0f}")
print("\nexamples (S1 name | true S2 name | true cos | K-th cos | top-3 retrieved):")
for e in examples:
    print(" ", e)
