"""Export compact text-pair files for training / scoring the cross-encoder elsewhere (e.g. Kaggle GPUs).

train : <ce-work>/pairs_pruned/train  ->  ce_train.parquet (text_a, text_b, label)
score : pruned pairs of a run whose pre-ranker probability lies in [lo, hi] -> score_<split>.parquet
        (s1_id, cand_id, text_a, text_b); only "uncertain" pairs need the cross-encoder.
"""
import argparse
import glob
import os

import pandas as pd

from .cross_encoder import raw_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=0.995)
    a = ap.parse_args()
    split = "train" if a.cmd == "train" else a.split
    files = sorted(glob.glob(os.path.join(a.work, "pairs_pruned", split, "[!_]*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id", "pre_p"]) for f in files], ignore_index=True)
    if a.cmd == "score":
        df = df[(df["pre_p"] >= a.lo) & (df["pre_p"] <= a.hi)].reset_index(drop=True)
    txt = raw_texts(a.data_dir, split, set(df["s1_id"]) | set(df["cand_id"]))
    df["text_a"] = [txt[s] for s in df["s1_id"]]
    df["text_b"] = [txt[c] for c in df["cand_id"]]
    if a.cmd == "train":
        gt = pd.read_parquet(os.path.join(a.work, "norm", "train_gt.parquet"))
        gt = {s: set(x.split(",")) if x else set() for s, x in zip(gt["s1_id"], gt["matched"])}
        df["label"] = [float(c in gt.get(s, ())) for s, c in zip(df["s1_id"], df["cand_id"])]
        df = df[["text_a", "text_b", "label"]]
    else:
        df = df[["s1_id", "cand_id", "text_a", "text_b"]]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_parquet(a.out, index=False, compression="zstd")
    print(f"wrote {len(df):,} rows -> {a.out} ({os.path.getsize(a.out) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
