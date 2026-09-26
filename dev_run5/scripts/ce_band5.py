"""Run-5 cross-encoder inputs.
  band  <work> <lo> <hi>   : pruned pairs with pre_p in [lo, hi] -> <work>/ce/band_<split>.parquet, and the subset
                             that has no e5 score yet in the run-4 score files -> <work>/ce/todo_<split>.parquet
  merge <work>             : run-4 scores (reused) + newly scored todo pairs -> <work>/ce/<split>.parquet
e5 scores depend only on the two raw records, so run-4 scores are reused for every pair they cover.
"""
import glob, os, sys
import pandas as pd

OLD = r"C:\Users\ANEXUS\Downloads\PsychicLearn_work4\ce"
cmd, W = sys.argv[1], sys.argv[2]
CE = os.path.join(W, "ce")
os.makedirs(CE, exist_ok=True)
for split in ("train", "test"):
    if cmd == "band":
        lo, hi = float(sys.argv[3]), float(sys.argv[4])
        parts = []
        for f in sorted(glob.glob(os.path.join(W, "pairs_pruned", split, "[!_]*.parquet"))):
            d = pd.read_parquet(f, columns=["s1_id", "cand_id", "pre_p"])
            parts.append(d[(d["pre_p"] >= lo) & (d["pre_p"] <= hi)][["s1_id", "cand_id"]])
        band = pd.concat(parts, ignore_index=True)
        band.to_parquet(os.path.join(CE, f"band_{split}.parquet"), index=False)
        old = pd.read_parquet(os.path.join(OLD, f"{split}.parquet"), columns=["s1_id", "cand_id"])
        todo = band.merge(old, on=["s1_id", "cand_id"], how="left", indicator=True)
        todo = todo[todo["_merge"] == "left_only"][["s1_id", "cand_id"]]
        todo.to_parquet(os.path.join(CE, f"todo_{split}.parquet"), index=False)
        print(f"{split}: band {len(band):,} pairs, already scored {len(band) - len(todo):,}, to score {len(todo):,}", flush=True)
    else:
        band = pd.read_parquet(os.path.join(CE, f"band_{split}.parquet"))
        old = pd.read_parquet(os.path.join(OLD, f"{split}.parquet"))
        new = pd.read_parquet(os.path.join(CE, f"scored_{split}.parquet"))[["s1_id", "cand_id", "ce_p"]]
        sc = pd.concat([old, new], ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        out = band.merge(sc, on=["s1_id", "cand_id"], how="left")
        assert out["ce_p"].notna().all(), f"{split}: {out['ce_p'].isna().sum()} band pairs without a score"
        out.to_parquet(os.path.join(CE, f"{split}.parquet"), index=False)
        print(f"{split}: {len(out):,} band pairs with e5 scores, mean {out['ce_p'].mean():.4f}", flush=True)
