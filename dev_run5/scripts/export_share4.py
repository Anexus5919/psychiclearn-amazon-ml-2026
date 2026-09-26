"""Run-4 files for teammates -> Downloads/PsychicLearn_share/from_L1 (uploaded to GitHub release from-L1).

L2: run4_oof.parquet (s1_id, cand_id, src, country, fold, p, label) + run4_truth.parquet
L3: run4_train_features.parquet (same keys + all model features) + features_run4.json
OOF p is rebuilt with the saved fold models (each row predicted by the model that did not train on it).
"""
import glob, json, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, r"C:\Users\ANEXUS\Downloads\PsychicLearn_dev\src")
from ber import decide, model, scoring

W = r"C:\Users\ANEXUS\Downloads\PsychicLearn_work4"
OUT = r"C:\Users\ANEXUS\Downloads\PsychicLearn_share\from_L1"
os.makedirs(OUT, exist_ok=True)
rep = json.load(open(W + r"\report_train.json"))
feats, t1, t2 = rep["main_inputs"]["features"], rep["t1"], rep["t2"]

parts = []
for f in sorted(glob.glob(W + r"\pairs_pruned\train\*.parquet")):
    d = pd.read_parquet(f, columns=["s1_id", "cand_id"] + feats)
    d.insert(2, "country", os.path.basename(f)[: -len(".parquet")])
    parts.append(d)
df = pd.concat(parts, ignore_index=True)
del parts
df[feats] = df[feats].astype(np.float32)
s2 = set(pd.read_parquet(W + r"\norm\train_s2.parquet", columns=["entity_id"])["entity_id"])
df.insert(2, "src", np.where(df["cand_id"].isin(s2), 2, 3).astype(np.int8))
df.insert(4, "fold", model.fold_ids(df["s1_id"].astype(str).values, 4))

gt = pd.read_parquet(W + r"\norm\train_gt.parquet")
gt = {s: set(x.split(",")) if x else set() for s, x in zip(gt["s1_id"], gt["matched"])}
queries = pd.read_parquet(W + r"\pairs\train\_queries.parquet")["s1_id"].tolist()
truth = {s: gt[s] for s in queries}
df.insert(5, "label", np.fromiter((c in truth[s] for s, c in zip(df["s1_id"], df["cand_id"])), np.int8, len(df)))

p = np.zeros(len(df), np.float32)
X = df[feats].to_numpy(np.float32)
for k in range(4):
    m = lgb.Booster(model_file=W + rf"\models\fold{k}.txt")
    idx = np.nonzero(df["fold"].values == k)[0]
    for s in range(0, len(idx), 500_000):
        r = idx[s:s + 500_000]
        p[r] = m.predict(X[r], num_iteration=m.best_iteration, num_threads=8)
del X
df.insert(6, "p", p)

f05 = scoring.macro_f05(decide.decide(df, t1, t2), truth)
print(f"check: OOF macro F0.5 = {f05:.5f} (report says {rep['oof_macro_f05']:.5f})", flush=True)

keys = ["s1_id", "cand_id", "src", "country", "fold", "label", "p"]
df[keys].to_parquet(OUT + r"\run4_oof.parquet", index=False, compression="zstd")
country = pd.read_parquet(W + r"\norm\train_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id")["country"]
cands = df.groupby("s1_id")["cand_id"].apply(set).to_dict()
pd.DataFrame({"s1_id": queries, "country": [country[s] for s in queries],
              "n_true": [len(truth[s]) for s in queries],
              "n_true_found": [len(truth[s] & cands.get(s, set())) for s in queries]}
             ).to_parquet(OUT + r"\run4_truth.parquet", index=False)
df[keys + feats].to_parquet(OUT + r"\run4_train_features.parquet", index=False, compression="zstd")
json.dump({"features": feats, "t1": t1, "t2": t2, "oof_macro_f05": f05, "lgb_params": model.PARAMS,
           "folds": "ber/model.py fold_ids(s1_id as str, 4 folds, seed 42); also stored in column 'fold'",
           "note": "p = out-of-fold LightGBM probability (run 4). Pairs are the pruned candidates only; "
                   "true matches that retrieval/pruning missed are counted in run4_truth.n_true."},
          open(OUT + r"\features_run4.json", "w"), indent=2)
for f in sorted(os.listdir(OUT)):
    print(f"{f}: {os.path.getsize(os.path.join(OUT, f)) / 1e6:.0f} MB", flush=True)
