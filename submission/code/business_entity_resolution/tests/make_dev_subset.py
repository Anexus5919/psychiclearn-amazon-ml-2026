"""Build a tiny, self-consistent copy of the dataset for fast end-to-end smoke tests.

Train: a sample of S1 entities per country, all of their ground-truth matches, plus random
distractor records. Test: a sample of S1 per country plus random pool records. Output has the
same folder/file layout as the real dataset.
"""
import argparse
import os

import numpy as np
import pandas as pd

from ber import io_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-s1", type=int, default=1500, help="S1 entities per country")
    ap.add_argument("--n-extra", type=int, default=4000, help="random extra pool records per source and country")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        os.makedirs(os.path.join(a.out_dir, split), exist_ok=True)
        s1 = io_utils.read_source(a.data_dir, split, 1)
        keep = []
        for c, g in s1.groupby("country"):
            keep.extend(g.index[rng.choice(len(g), min(a.n_s1, len(g)), replace=False)])
        s1 = s1.loc[sorted(keep)]
        need = set()
        if split == "train":
            gt = io_utils.read_ground_truth(a.data_dir)
            gt = {k: gt[k] for k in s1["entity_id"]}
            for v in gt.values():
                need |= v
            with open(os.path.join(a.out_dir, split, "train_ground_truth.tsv"), "w", encoding="utf-8", newline="\n") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                for k, v in gt.items():
                    f.write(f"{k}\t{','.join(sorted(v))}\n")
        s1.to_csv(os.path.join(a.out_dir, split, f"{split}_source1.tsv"), sep="\t", index=False, lineterminator="\n")
        for src in (2, 3):
            pool = io_utils.read_source(a.data_dir, split, src)
            mask = pool["entity_id"].isin(need).values
            for c, g in pool.groupby("country"):
                mask[g.index[rng.choice(len(g), min(a.n_extra, len(g)), replace=False)]] = True
            pool[mask].to_csv(os.path.join(a.out_dir, split, f"{split}_source{src}.tsv"), sep="\t", index=False,
                              lineterminator="\n")
        print(split, "done")


if __name__ == "__main__":
    main()
