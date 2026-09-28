"""The final set of training/validation Source-1 businesses (304,555 in our run).

How the set was built (three disjoint samples of the TRAIN Source-1 businesses, stratified by country):
  * base set   : `pipeline.train_query_ids` (train_frac 0.08, seed 42), written by the base run's `pairs` stage.
                 The Indic->Latin transliteration dictionary is learned with these businesses excluded.
  * CE set     : 3% of the remaining businesses per country (seed 123), written by `ber.ce_data`. Only the
                 cross-encoder is trained on them, so its scores can be stacked into LightGBM without leakage.
  * final set  : base set + 6.5% of the businesses in neither set (seed 2024). LightGBM trains/validates on
                 these (4-fold grouped CV), and they are excluded from every other learned component
                 (translit dictionary for the base part, region map, dense bi-encoder, cross-encoders).

Usage: python -m ber.queries --base <base work dir> --ce <ce work dir> --final <final work dir>
Copies <base>/norm/* to <final>/norm/ and writes <final>/pairs/train/_queries.parquet.
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ce", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--frac", type=float, default=0.065)
    ap.add_argument("--seed", type=int, default=2024)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.final, "norm"), exist_ok=True)
    os.makedirs(os.path.join(a.final, "pairs", "train"), exist_ok=True)
    for f in os.listdir(os.path.join(a.base, "norm")):
        dst = os.path.join(a.final, "norm", f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(a.base, "norm", f), dst)
    out = os.path.join(a.final, "pairs", "train", "_queries.parquet")
    if os.path.exists(out):
        print(f"exists: {out}")
        return
    q_base = set(pd.read_parquet(os.path.join(a.base, "pairs", "train", "_queries.parquet"))["s1_id"])
    q_ce = set(pd.read_parquet(os.path.join(a.ce, "pairs", "train", "_queries.parquet"))["s1_id"])
    s1 = pd.read_parquet(os.path.join(a.final, "norm", "train_s1.parquet"), columns=["entity_id", "country"])
    free = s1[~s1["entity_id"].isin(q_base | q_ce)]
    rng = np.random.default_rng(a.seed)
    extra = []
    for _, g in free.groupby("country"):
        n = int(round(len(g) * a.frac))
        extra.extend(g["entity_id"].values[rng.choice(len(g), n, replace=False)])
    q = sorted(q_base | set(extra))
    assert not (set(q) & q_ce), "final set must be disjoint from the cross-encoder businesses"
    pd.DataFrame({"s1_id": q}).to_parquet(out, index=False)
    print(f"final training/validation businesses: {len(q):,} (base {len(q_base):,} + {len(extra):,}); "
          f"cross-encoder businesses excluded: {len(q_ce):,}")


if __name__ == "__main__":
    main()
