"""Run-4 error breakdown: where is F0.5 lost (retrieval / pruning / model FN / FP), per country, with examples."""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\ANEXUS\Downloads\PsychicLearn_dev\src")
from ber import decide, scoring
from ber.cross_encoder import raw_texts

W = r"C:\Users\ANEXUS\Downloads\PsychicLearn_work4"
DATA = r"C:\Users\ANEXUS\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
SH = r"C:\Users\ANEXUS\Downloads\PsychicLearn_share\from_L1"
oof = pd.read_parquet(SH + r"\run4_oof.parquet")
ce = pd.read_parquet(W + r"\ce\train.parquet")
oof = oof.merge(ce, on=["s1_id", "cand_id"], how="left")
gt = pd.read_parquet(W + r"\norm\train_gt.parquet")
gt = {s: set(x.split(",")) if x else set() for s, x in zip(gt["s1_id"], gt["matched"])}
queries = pd.read_parquet(W + r"\pairs\train\_queries.parquet")["s1_id"].tolist()
truth = {s: gt[s] for s in queries}
country = pd.read_parquet(W + r"\norm\train_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id")["country"]
pred = decide.decide(oof, 0.54, 0.74)
cands = oof.groupby("s1_id")["cand_id"].apply(set).to_dict()
raw = set()
for f in glob.glob(W + r"\pairs\train\*_s*.parquet"):
    d = pd.read_parquet(f, columns=["s1_id", "cand_id"])
    raw |= set(zip(d["s1_id"], d["cand_id"]))
print("raw retrieved pairs:", len(raw))

def f_of(pm, ids):
    return np.mean([scoring.f05_entity(set(pm.get(s, ())), truth[s]) for s in ids])

rows = []
for c in ("India", "US"):
    ids = [s for s in queries if country[s] == c]
    tp_all = sum(len(truth[s]) for s in ids)
    miss_ret = [(s, t) for s in ids for t in truth[s] if (s, t) not in raw]
    miss_prn = [(s, t) for s in ids for t in truth[s] if (s, t) in raw and t not in cands.get(s, ())]
    base = f_of(pred, ids)
    no_fp = f_of({s: [x for x in pred.get(s, []) if x in truth[s]] for s in ids}, ids)
    no_fn = f_of({s: list(set(pred.get(s, [])) | (truth[s] & cands.get(s, set()))) for s in ids}, ids)
    print(f"\n== {c}: n={len(ids):,} true pairs={tp_all:,}  F0.5={base:.5f}")
    print(f"   retrieval misses {len(miss_ret):,} ({len(miss_ret)/tp_all:.2%})   pruning misses {len(miss_prn):,} ({len(miss_prn)/tp_all:.2%})")
    print(f"   remove all FP -> {no_fp:.5f} (+{no_fp-base:.4f})   add all model-missed (in cands) -> {no_fn:.5f} (+{no_fn-base:.4f})")
    sub = oof[oof["s1_id"].isin(set(ids))]
    pm = {(s, x) for s in ids for x in pred.get(s, [])}
    key = list(zip(sub["s1_id"], sub["cand_id"]))
    isp = np.array([k in pm for k in key])
    fn = sub[(sub["label"] == 1) & ~isp]
    fp = sub[(sub["label"] == 0) & isp]
    bins = [0, 0.02, 0.1, 0.3, 0.54, 0.74, 1.01]
    print("   model FN by p:", pd.cut(fn["p"], bins).value_counts().sort_index().to_dict())
    print("   model FN with ce_p (in band):", int(fn["ce_p"].notna().sum()), "of", len(fn), "| FN ce_p>0.5:", int((fn["ce_p"] > 0.5).sum()))
    print("   FP by p:", pd.cut(fp["p"], bins).value_counts().sort_index().to_dict(), "| FP total", len(fp))
    rows.append((c, miss_ret, fn, fp))

# text examples: India retrieval misses, India model FN, India FP
ex_ids = set()
rng = np.random.default_rng(0)
c, miss_ret, fn, fp = rows[0]
smp_ret = [miss_ret[i] for i in rng.choice(len(miss_ret), 12, replace=False)]
smp_fn = fn.sample(8, random_state=0)[["s1_id", "cand_id", "p", "ce_p"]].values.tolist()
smp_fp = fp.sample(8, random_state=0)[["s1_id", "cand_id", "p", "ce_p"]].values.tolist()
for s, t in smp_ret: ex_ids |= {s, t}
for r in smp_fn + smp_fp: ex_ids |= {r[0], r[1]}
txt = raw_texts(DATA, "train", ex_ids)
print("\n--- India RETRIEVAL misses (S1 || true match) ---")
for s, t in smp_ret: print(f"  {txt[s][:110]}\n    || {txt[t][:110]}")
print("\n--- India MODEL misses (in candidates, not predicted) ---")
for s, t, p, cp in smp_fn: print(f"  p={p:.3f} ce={cp}\n  {txt[s][:110]}\n    || {txt[t][:110]}")
print("\n--- India FALSE merges ---")
for s, t, p, cp in smp_fp: print(f"  p={p:.3f} ce={cp}\n  {txt[s][:110]}\n    || {txt[t][:110]}")
