"""LightGBM pair classifier with grouped out-of-fold training.

Folds are grouped by Source-1 entity so every candidate of an entity is scored by a model that
never saw that entity; the out-of-fold probabilities drive threshold tuning and the reported
validation score.
"""
import lightgbm as lgb
import numpy as np

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.08,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbose": -1,
    "seed": 42,
}


def fold_ids(groups, n_folds, seed=42):
    """Assign each distinct group (S1 entity) to a fold, deterministically."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = dict(zip(uniq[perm], np.arange(len(uniq)) % n_folds))
    return np.array([fold_of[g] for g in groups], dtype=np.int8)


def train_oof(X, y, groups, n_folds, n_jobs, rounds=3000, log=print):
    """Train one model per fold; return (oof_probabilities, list_of_boosters)."""
    params = dict(PARAMS, num_threads=n_jobs)
    folds = fold_ids(groups, n_folds)
    oof = np.zeros(len(y), dtype=np.float32)
    models = []
    full = lgb.Dataset(X, label=y, params={"max_bin": params["max_bin"]}, free_raw_data=False).construct()
    for k in range(n_folds):
        tr, va = np.nonzero(folds != k)[0], np.nonzero(folds == k)[0]
        dtr, dva = full.subset(tr), full.subset(va)  # share one binned dataset; no raw copies
        bst = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        for s in range(0, len(va), 1_000_000):
            rows = va[s:s + 1_000_000]
            oof[rows] = bst.predict(X[rows], num_iteration=bst.best_iteration)
        log(f"    fold {k}: best_iter={bst.best_iteration} valid_logloss={bst.best_score['valid_0']['binary_logloss']:.5f}")
        models.append(bst)
    return oof, models


def predict(models, X, chunk=2_000_000, n_jobs=0):
    """Average the fold models' probabilities, in row chunks to bound memory."""
    out = np.zeros(X.shape[0], dtype=np.float32)
    for s in range(0, X.shape[0], chunk):
        xs = X[s:s + chunk]
        out[s:s + chunk] = np.mean([m.predict(xs, num_iteration=m.best_iteration, num_threads=n_jobs) for m in models], axis=0)
    return out
