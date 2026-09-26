"""Write the cross-encoder band files: pruned pairs with pre_p in [0.02, 0.995] -> work4/ce/band_<split>_<country>.parquet.
Usage: ce_band.py <split> <country> [<country> ...]"""
import os, sys
import pandas as pd

W4 = r"C:\Users\ANEXUS\Downloads\PsychicLearn_work4"
os.makedirs(os.path.join(W4, "ce"), exist_ok=True)
split = sys.argv[1]
for country in sys.argv[2:]:
    df = pd.read_parquet(os.path.join(W4, "pairs_pruned", split, f"{country}.parquet"), columns=["s1_id", "cand_id", "pre_p"])
    df = df[(df["pre_p"] >= 0.02) & (df["pre_p"] <= 0.995)][["s1_id", "cand_id"]]
    df.to_parquet(os.path.join(W4, "ce", f"band_{split}_{country}.parquet"), index=False)
    print(f"band {split} {country}: {len(df):,} pairs", flush=True)
