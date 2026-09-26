"""Does pseudo-labelling help a country with no labels? Proxy on run-4 features: India = labelled,
US = 'unlabelled target'. A: train on India only -> predict US. B/C: add confident US pseudo-labels,
retrain, predict US again. Score each against the real US labels (macro F0.5 incl. retrieval misses)."""
import sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"C:\Users\ANEXUS\Downloads\PsychicLearn_dev\src")
from ber import decide, scoring, model

SH = r"C:\Users\ANEXUS\Downloads\PsychicLearn_share\from_L1"
import json
meta = json.load(open(SH + r"\features_run4.json"))
feats = meta["features"]
import pyarrow.parquet as pq
ents = pd.read_parquet(SH + r"\run4_truth.parquet", columns=["s1_id"])["s1_id"].values
rng = np.random.default_rng(0)
keep = set(ents[rng.random(len(ents)) < 0.5])  # 50% of businesses (RAM: the pairs stage runs in parallel)
df = pq.read_table(SH + r"\run4_train_features.parquet", filters=[("s1_id", "in", list(keep))]).to_pandas()
tr = pd.read_parquet(SH + r"\run4_truth.parquet")
tr = tr[tr["s1_id"].isin(keep)]
found = df[df["label"] == 1].groupby("s1_id")["cand_id"].apply(set).to_dict()
truth = {}
for s, n in zip(tr["s1_id"], tr["n_true"]):
    t = set(found.get(s, ()))
    truth[s] = t | {f"__miss_{s}_{i}" for i in range(int(n) - len(t))}
us_ids = tr.loc[tr["country"] == "US", "s1_id"].tolist()
ind = df["country"].values == "India"
us = ~ind
P = dict(model.PARAMS, num_threads=4, verbose=-1)
T1, T2 = meta["t1"], meta["t2"]


def fit(X, y, rounds=600):
    return lgb.train(P, lgb.Dataset(X, label=y, free_raw_data=True), num_boost_round=rounds)


def f05_us(p):
    d = df.loc[us, ["s1_id", "cand_id"]].assign(p=p)
    pred = decide.decide(d, T1, T2)
    return np.mean([scoring.f05_entity(set(pred.get(s, ())), truth[s]) for s in us_ids]), d, pred


X_all = df[feats].to_numpy(np.float32)
y_all = df["label"].values
print(f"rows: India {ind.sum():,}  US {us.sum():,}", flush=True)
mA = fit(X_all[ind], y_all[ind])
pA = mA.predict(X_all[us])
fA, dA, predA = f05_us(pA)
print(f"A  India-only model on US:              F0.5 = {fA:.5f}", flush=True)
# pseudo-labels from A: positives = accepted pairs with p >= hi; negatives = pairs with p <= lo
acc = {(s, c) for s, cs in predA.items() for c in cs}
is_acc = np.fromiter(((s, c) in acc for s, c in zip(dA["s1_id"], dA["cand_id"])), bool, len(dA))
for name, hi, lo in (("B", 0.95, 0.02), ("C", 0.90, 0.05)):
    pos, neg = is_acc & (pA >= hi), pA <= lo
    sel = pos | neg
    Xp, yp = X_all[us][sel], pos[sel].astype(np.int8)
    print(f"{name} pseudo-labels: {int(pos.sum()):,} positives (true-label precision {y_all[us][pos].mean():.4f}), "
          f"{int(neg.sum()):,} negatives (true-label NPV {1 - y_all[us][neg].mean():.4f})", flush=True)
    mB = fit(np.vstack([X_all[ind], Xp]), np.concatenate([y_all[ind], yp]))
    fB, _, _ = f05_us(mB.predict(X_all[us]))
    print(f"{name}  India + US pseudo-labels, on US:     F0.5 = {fB:.5f}  ({fB - fA:+.5f} vs A)", flush=True)
mU = fit(X_all[us], y_all[us])  # oracle: real US labels (in-sample, upper bound only)
print(f"ref US-labelled model (in-sample, upper bound): F0.5 = {f05_us(mU.predict(X_all[us]))[0]:.5f}", flush=True)
