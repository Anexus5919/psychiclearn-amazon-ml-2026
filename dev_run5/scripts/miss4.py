"""Why are run-4 India/US retrieval misses missed? Name/address similarity of missed true pairs after
our normalisation, vs found true pairs; composition by noise type."""
import glob, re, sys
import numpy as np, pandas as pd
from rapidfuzz import fuzz
sys.path.insert(0, r"C:\Users\ANEXUS\Downloads\PsychicLearn_dev\src")

W = r"C:\Users\ANEXUS\Downloads\PsychicLearn_work4"
gt = pd.read_parquet(W + r"\norm\train_gt.parquet")
gt = {s: x.split(",") if x else [] for s, x in zip(gt["s1_id"], gt["matched"])}
queries = pd.read_parquet(W + r"\pairs\train\_queries.parquet")["s1_id"].tolist()
raw = set()
for f in glob.glob(W + r"\pairs\train\*_s*.parquet"):
    d = pd.read_parquet(f, columns=["s1_id", "cand_id"])
    raw |= set(zip(d["s1_id"], d["cand_id"]))
cols = ["entity_id", "country", "name_core", "addr_core", "name_native", "addr_empty"]
need = set(queries) | {t for s in queries for t in gt[s]}
parts = []
for i in (1, 2, 3):
    d = pd.read_parquet(W + rf"\norm\train_s{i}.parquet", columns=cols)
    parts.append(d[d["entity_id"].isin(need)])
    del d
norm = pd.concat(parts).set_index("entity_id")
del parts
rng = np.random.default_rng(1)
for c in ("India", "US"):
    qs = [s for s in queries if norm.at[s, "country"] == c]
    pairs = [(s, t) for s in qs for t in gt[s]]
    miss = [p for p in pairs if p not in raw]
    found = [pairs[i] for i in rng.choice(len(pairs), 20000, replace=False)]
    found = [p for p in found if p in raw]
    def stats(ps):
        a = norm.loc[[p[0] for p in ps]]
        b = norm.loc[[p[1] for p in ps]]
        ns = np.array([fuzz.token_set_ratio(x, y) for x, y in zip(a["name_core"], b["name_core"])])
        as_ = np.array([fuzz.token_set_ratio(x, y) for x, y in zip(a["addr_core"], b["addr_core"])])
        return ns, as_, b["name_native"].values, b["addr_empty"].values
    for label, ps in (("MISSED", miss), ("found", found)):
        ns, as_, nat, emp = stats(ps)
        print(f"{c} {label:6s} n={len(ps):,}  name_sim>=90: {np.mean(ns>=90):.1%}  name_sim<60: {np.mean(ns<60):.1%}  "
              f"addr_sim>=80: {np.mean(as_>=80):.1%}  cand native-script: {np.mean(nat):.1%}  cand addr empty: {np.mean(emp):.1%}")
    ns, as_, nat, emp = stats(miss)
    a = norm.loc[[p[0] for p in miss]]; b = norm.loc[[p[1] for p in miss]]
    print(f"  {c} missed with HIGH name sim (>=90) -> crowding: {np.mean(ns>=90):.1%}; LOW name sim (<60): {np.mean(ns<60):.1%}")
    idx = np.nonzero(ns < 60)[0]
    for i in rng.choice(idx, min(10, len(idx)), replace=False):
        print(f"    low-sim: [{a['name_core'].iat[i]}] vs [{b['name_core'].iat[i]}] | addr [{a['addr_core'].iat[i][:50]}] vs [{b['addr_core'].iat[i][:50]}]")
    idx = np.nonzero(ns >= 90)[0]
    for i in rng.choice(idx, min(6, len(idx)), replace=False):
        print(f"    high-sim: [{a['name_core'].iat[i]}] vs [{b['name_core'].iat[i]}] | addr [{a['addr_core'].iat[i][:50]}] vs [{b['addr_core'].iat[i][:50]}]")
