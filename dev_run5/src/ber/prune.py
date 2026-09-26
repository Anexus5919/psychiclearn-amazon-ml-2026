"""Learned candidate pruning (second, final stage of candidate generation).

The retrieval passes return ~50 candidates per Source-1 entity. A small LightGBM "pre-ranker"
using only cheap retrieval-derived signals (TF-IDF cosines, per-pass ranks, per-entity context,
name/address frequencies) scores them, and each entity keeps only its top candidates. The keep
rule (top-N and a probability floor) is chosen on out-of-fold predictions as the smallest
candidate set that loses at most `max_recall_loss` of the true pairs the retrieval found.
The pruned set is exactly what the main matcher scores, and is written to candidate_pairs.tsv.
"""
import itertools

import lightgbm as lgb
import numpy as np

CHEAP = ["cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "rank_region", "region_match",
         "ctx_rank_combo_src",
         "ctx_gap_combo", "ctx_ncand", "ctx_rank_name_all", "ctx_rank_addr_all", "ctx_rank_combo_all",
         "freq_s1_name", "freq_pool_name", "freq_s1_addr", "src_is_s3", "p_addr_empty", "p_native", "p_handle"]
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 7}


def train_oof(X, y, folds, n_jobs, rounds=400):
    """OOF pre-ranker probabilities plus one model per fold."""
    params = dict(PARAMS, num_threads=n_jobs)
    full = lgb.Dataset(X, label=y, free_raw_data=False).construct()
    oof = np.zeros(len(y), np.float32)
    models = []
    for k in np.unique(folds):
        tr, va = np.nonzero(folds != k)[0], np.nonzero(folds == k)[0]
        m = lgb.train(params, full.subset(tr), num_boost_round=rounds, valid_sets=[full.subset(va)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        models.append(m)
    return oof, models


def within_entity_rank(s1_codes, p):
    """1-based rank of each row's p within its S1 entity (descending)."""
    order = np.lexsort((-p, s1_codes))
    ranks = np.empty(len(p), np.int32)
    s_sorted = s1_codes[order]
    starts = np.r_[0, np.nonzero(np.diff(s_sorted))[0] + 1]
    r = np.arange(len(p)) - np.repeat(starts, np.diff(np.r_[starts, len(p)]))
    ranks[order] = r + 1
    return ranks


def choose_rule(p, rank, label, n_entities, max_recall_loss,
                n_grid=(3, 4, 5, 6, 8, 10, 12, 15, 20, 30), t_grid=(0.0, 0.0005, 0.001, 0.003, 0.01, 0.02, 0.05)):
    """Smallest mean candidates/entity whose kept true pairs >= (1 - max_recall_loss) of all found.
    Also returns the size/recall frontier for several loss budgets (reported in the log)."""
    total_true = max(1, label.sum())
    grid = []
    for n, t in itertools.product(n_grid, t_grid):
        keep = (rank <= n) & (p >= t)
        grid.append({"top_n": int(n), "min_p": float(t), "recall_loss": float(1 - (label & keep).sum() / total_true),
                     "per_entity": float(keep.sum() / n_entities)})

    def best_within(budget):
        ok = [g for g in grid if g["recall_loss"] <= budget]
        return min(ok, key=lambda g: g["per_entity"]) if ok else max(grid, key=lambda g: g["per_entity"])

    rule = dict(best_within(max_recall_loss))
    rule["frontier"] = {str(b): {k: round(v, 5) for k, v in best_within(b).items()} for b in (0.0005, 0.001, 0.002, 0.005, 0.01)}
    return rule
