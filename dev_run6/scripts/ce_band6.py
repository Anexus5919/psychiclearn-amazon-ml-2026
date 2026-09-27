"""Run-6 cross-encoder inputs (same as ce_band5.py, but reuses the e5 scores of runs 4 AND 5).
  band  <work> <lo> <hi>   : pruned pairs with pre_p in [lo, hi] -> <work>/ce/band_<split>.parquet, and the subset
                             without an existing e5 score -> <work>/ce/todo_<split>.parquet
  merge <work>             : reused scores + newly scored todo pairs -> <work>/ce/<split>.parquet
e5 scores depend only on the two raw records, so earlier scores are valid for every pair they cover.
"""
import glob, os, sys
import pandas as pd

OLDS = [
    d for d in [
        r"C:\Users\ANEXUS\Downloads\PsychicLearn_work4\ce",
        r"C:\Users\ANEXUS\Downloads\PsychicLearn_work5\ce",
        os.path.join(os.path.dirname(W), "PsychicLearn_work4", "ce"),
        os.path.join(os.path.dirname(W), "PsychicLearn_work5", "ce")
    ] if os.path.exists(d)
]


def old_scores(split):
    parts = []
    for d in OLDS:
        fp = os.path.join(d, f"{split}.parquet")
        if os.path.exists(fp):
            parts.append(pd.read_parquet(fp, columns=["s1_id", "cand_id", "ce_p"]))
    if not parts:
        return pd.DataFrame(columns=["s1_id", "cand_id", "ce_p"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])


for split in ("train", "test"):
    if cmd == "band":
        lo, hi = float(sys.argv[3]), float(sys.argv[4])
        parts = []
        for f in sorted(glob.glob(os.path.join(W, "pairs_pruned", split, "[!_]*.parquet"))):
            d = pd.read_parquet(f, columns=["s1_id", "cand_id", "pre_p"])
            parts.append(d[(d["pre_p"] >= lo) & (d["pre_p"] <= hi)][["s1_id", "cand_id"]])
        band = pd.concat(parts, ignore_index=True).drop_duplicates(["s1_id", "cand_id"]) if parts else pd.DataFrame(columns=["s1_id", "cand_id"])
        band.to_parquet(os.path.join(CE, f"band_{split}.parquet"), index=False)
        old = old_scores(split)
        if len(old):
            todo = band.merge(old[["s1_id", "cand_id"]], on=["s1_id", "cand_id"], how="left", indicator=True)
            todo = todo[todo["_merge"] == "left_only"][["s1_id", "cand_id"]]
        else:
            todo = band
        todo.to_parquet(os.path.join(CE, f"todo_{split}.parquet"), index=False)
        print(f"{split}: band {len(band):,} pairs, already scored {len(band) - len(todo):,}, to score {len(todo):,}", flush=True)
    else:
        band = pd.read_parquet(os.path.join(CE, f"band_{split}.parquet"))
        new = pd.read_parquet(os.path.join(CE, f"scored_{split}.parquet"))[["s1_id", "cand_id", "ce_p"]]
        sc = pd.concat([old_scores(split), new], ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        out = band.merge(sc, on=["s1_id", "cand_id"], how="left")
        assert out["ce_p"].notna().all(), f"{split}: {out['ce_p'].isna().sum()} band pairs without a score"
        out.to_parquet(os.path.join(CE, f"{split}.parquet"), index=False)
        print(f"{split}: {len(out):,} band pairs with e5 scores, mean {out['ce_p'].mean():.4f}", flush=True)
