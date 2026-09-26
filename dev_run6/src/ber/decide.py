"""Turning pair probabilities into per-entity match sets.

1. Exclusive assignment: in the training ground truth every S2/S3 record belongs to at most one
   S1 entity, so each candidate record is kept only for the S1 entity that scores it highest.
2. Rank-dependent thresholds: under F0.5 the first match of an entity is worth accepting at a much
   lower probability than later ones (blueprint §1.11), so rank 1 uses t1 and ranks >= 2 use t2.
"""
import itertools

import numpy as np


def _assign_and_rank(df, exclusive=True):
    """Apply exclusive assignment, then rank each S1 entity's surviving candidates by p."""
    d = df
    if exclusive:
        d = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("cand_id", keep="first")
    d = d.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    d["rank"] = d.groupby("s1_id").cumcount().values + 1
    return d


def decide(df, t1, t2, exclusive=True):
    """df columns: s1_id, cand_id, p. Returns {s1_id: [cand_id, ...]} for accepted pairs."""
    d = _assign_and_rank(df[["s1_id", "cand_id", "p"]], exclusive)
    keep = np.where(d["rank"].values == 1, d["p"].values >= t1, d["p"].values >= t2)
    return d[keep].groupby("s1_id")["cand_id"].apply(list).to_dict()


def tune_thresholds(df, truth, grid1=None, grid2=None, exclusive=True):
    """Grid-search (t1, t2) maximising macro F0.5 over all S1 entities in `truth`.

    df needs columns s1_id, cand_id, p, label (1 if the pair is a true match). Entities in `truth`
    without any candidate are scored as empty predictions, and true matches missed by blocking
    count as false negatives, so the reported score is end-to-end.
    Returns (best_f05, t1, t2).
    """
    grid1 = grid1 if grid1 is not None else np.round(np.arange(0.02, 0.96, 0.02), 3)
    grid2 = grid2 if grid2 is not None else np.round(np.arange(0.10, 0.99, 0.02), 3)
    ents = list(truth.keys())
    code = {e: i for i, e in enumerate(ents)}
    n = len(ents)
    true_cnt = np.array([len(truth[e]) for e in ents], dtype=np.float64)
    d = _assign_and_rank(df[["s1_id", "cand_id", "p", "label"]], exclusive)
    d = d[d["s1_id"].isin(code)]
    ent = d["s1_id"].map(code).values
    rank1 = d["rank"].values == 1
    p = d["p"].values
    lab = d["label"].values.astype(bool)
    best = (-1.0, None, None)
    for t1, t2 in itertools.product(grid1, grid2):
        if t2 < t1:
            continue
        keep = np.where(rank1, p >= t1, p >= t2)
        pred = np.bincount(ent, weights=keep, minlength=n)
        tp = np.bincount(ent, weights=keep & lab, minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1.25 * tp / (1.25 * tp + 0.25 * (true_cnt - tp) + (pred - tp))
        f = np.where(true_cnt == 0, (pred == 0).astype(float), np.where(tp == 0, 0.0, f))
        score = float(f.mean())
        if score > best[0]:
            best = (score, float(t1), float(t2))
    return best
