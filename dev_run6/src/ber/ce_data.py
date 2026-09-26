"""Training data for the cross-encoder, from Source-1 entities DISJOINT from the LightGBM queries.

The cross-encoder must not be trained on the entities whose pairs LightGBM trains on; otherwise its
scores (used as a LightGBM feature) would leak labels. We therefore sample a fresh set of training
entities, run the same retrieval + pruning for them, and use those pairs (with ground-truth labels)
to fine-tune the cross-encoder.

Usage: python -m ber.ce_data --data-dir <dataset> --src-work <run-3 work dir> --ce-work <new dir>
"""
import argparse
import json
import os
import shutil

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import io_utils, pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--src-work", required=True, help="work dir of the main run (norm files, queries, pre-ranker)")
    ap.add_argument("--ce-work", required=True)
    ap.add_argument("--frac", type=float, default=0.03, help="share of each country's train S1 entities")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()

    norm_dst = io_utils.ensure_dir(os.path.join(a.ce_work, "norm"))
    for f in ("train_s1.parquet", "train_s2.parquet", "train_s3.parquet", "train_gt.parquet", "translit.json"):
        if not os.path.exists(os.path.join(norm_dst, f)):
            shutil.copy2(os.path.join(a.src_work, "norm", f), norm_dst)
    qdir = io_utils.ensure_dir(os.path.join(a.ce_work, "pairs", "train"))
    qpath = os.path.join(qdir, "_queries.parquet")
    if not os.path.exists(qpath):
        used = set(pd.read_parquet(os.path.join(a.src_work, "pairs", "train", "_queries.parquet"))["s1_id"])
        s1 = pd.read_parquet(os.path.join(norm_dst, "train_s1.parquet"), columns=["entity_id", "country"])
        s1 = s1[~s1["entity_id"].isin(used)]
        rng = np.random.default_rng(a.seed)
        ids = []
        for _, g in s1.groupby("country"):
            n = int(round(len(g) * a.frac))
            ids.extend(g["entity_id"].values[rng.choice(len(g), n, replace=False)])
        pd.DataFrame({"s1_id": sorted(ids)}).to_parquet(qpath, index=False)
    cfg = pipeline.Config(a.data_dir, a.ce_work, os.path.join(a.ce_work, "out"))
    pipeline.log(f"ce_data: {len(pd.read_parquet(qpath)):,} disjoint query entities", cfg)
    pipeline.stage_pairs(cfg, splits=("train",))
    rule = json.load(open(os.path.join(a.src_work, "prune", "rule.json")))
    pre = [lgb.Booster(model_file=os.path.join(a.src_work, "prune", "pre0.txt"))]
    pipeline._write_pruned(cfg, "train", rule, models=pre)
    pipeline.log("ce_data done", cfg)


if __name__ == "__main__":
    main()
