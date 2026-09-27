"""Gate for run 6 (memory-light): of the true matches that run-5 retrieval MISSED, how many does dense top-15 find?
Validation businesses only (excluded from the dense model's fine-tuning -> honest). Usage: gate_dense.py <dense_dir>"""
import glob, sys
import pandas as pd

W5 = r"C:\Users\ANEXUS\Downloads\PsychicLearn_work5"
dense = pd.read_parquet(sys.argv[1] + r"\dense_train.parquet", columns=["s1_id", "cand_id"])
gt = pd.read_parquet(W5 + r"\norm\train_gt.parquet")
q = set(pd.read_parquet(W5 + r"\pairs\train\_queries.parquet")["s1_id"])
gt = gt[gt["s1_id"].isin(q) & (gt["matched"] != "")]
true = gt.assign(cand_id=gt["matched"].str.split(",")).explode("cand_id")[["s1_id", "cand_id"]].reset_index(drop=True)
country = pd.read_parquet(W5 + r"\norm\train_s1.parquet", columns=["entity_id", "country"]).set_index("entity_id")["country"]
true["country"] = true["s1_id"].map(country).values
true["retrieved"] = False
dense_in_raw = 0
for f in glob.glob(W5 + r"\pairs\train\*_s*.parquet"):
    d = pd.read_parquet(f, columns=["s1_id", "cand_id"])
    m = true.merge(d.assign(hit=True), on=["s1_id", "cand_id"], how="left")["hit"].fillna(False).values.astype(bool)
    true["retrieved"] |= m
    dense_in_raw += len(dense.merge(d, on=["s1_id", "cand_id"]))
    del d
true["dense"] = true.merge(dense.assign(hit=True), on=["s1_id", "cand_id"], how="left")["hit"].fillna(False).values.astype(bool)
for c, g in true.groupby("country"):
    n, miss = len(g), g[~g["retrieved"]]
    found = int(miss["dense"].sum())
    print(f"{c}: true pairs {n:,} | run-5 retrieval recall {g['retrieved'].mean():.4f} | dense top-15 alone {g['dense'].mean():.4f} | "
          f"run-5 MISSED {len(miss):,} -> dense finds {found:,} ({found / max(1, len(miss)):.1%}) | union recall "
          f"{(g['retrieved'] | g['dense']).mean():.4f}", flush=True)
print(f"dense pairs {len(dense):,}, of which already retrieved by run 5: {dense_in_raw:,} -> new pairs {len(dense) - dense_in_raw:,}", flush=True)
