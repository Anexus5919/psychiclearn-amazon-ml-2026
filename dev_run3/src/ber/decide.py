"""Turning pair probabilities into per-entity match sets.

1. Exclusive assignment: in the training ground truth every S2/S3 record belongs to at most one
   S1 entity, so each candidate record is kept only for the S1 entity that scores it highest.
2. Rank-dependent thresholds: under F0.5 the first match of an entity is worth accepting at a much
   lower probability than later ones (blueprint §1.11), so rank 1 uses t1 and ranks >= 2 use t2.

L2 additions (feat/decision):
3. decide_v2: per-source thresholds (S2/S3), optional rank-3+ threshold, expected-F0.5 selection,
   and probability-sum rule.  All methods are exposed through a single params dict so that L1 can
   drop in the best params from decision_params.json without touching any other code.
4. tune_v2: half-business cross-validated tuning that prevents leakage.
"""
import itertools

import numpy as np


# ------------------------------------------------------------------ helpers

def _assign_and_rank(df, exclusive=True):
    """Apply exclusive assignment, then rank each S1 entity's surviving candidates by p."""
    d = df
    if exclusive:
        d = d.sort_values("p", ascending=False, kind="stable").drop_duplicates("cand_id", keep="first")
    d = d.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    d["rank"] = d.groupby("s1_id").cumcount().values + 1
    return d


# ------------------------------------------------------------------ v1 (unchanged – L1/L3/L4 must not be broken)

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


# ================================================================== L2: decide_v2
#
# params dict keys (all optional; defaults fall back to v1 behaviour):
#
#   method       str   "threshold" | "expected_f05" | "prob_sum"   (default "threshold")
#   exclusive    bool  apply exclusive assignment before thresholding (default True)
#
#   # --- threshold method ---
#   t1_s2        float threshold for S2, rank 1         (required for "threshold")
#   t2_s2        float threshold for S2, rank >= 2
#   t1_s3        float threshold for S3, rank 1         (falls back to t1_s2 if absent)
#   t2_s3        float threshold for S3, rank >= 2      (falls back to t2_s2 if absent)
#   t3           float threshold for ALL sources, rank >= 3  (falls back to t2_sx if absent)
#
#   # --- expected_f05 method ---
#   n_true_est   float prior estimate of n_true per entity (used for missed matches;
#                       if absent, the column "n_true_found" in df is used if available,
#                       otherwise a fixed value of 3.7 is used)
#   max_k        int   cap on selections per entity (default 30)
#
#   # --- prob_sum method ---
#   ps_t1        float minimum p for rank-1 acceptance (default 0.02)
#   ps_tmin      float minimum p for any acceptance (default 0.10)
#   ps_cap       int   max selections per entity (default 30)

def decide_v2(df, params):
    """Generalised decision function. df columns: s1_id, cand_id, src (2|3), p.

    Returns {s1_id: [cand_id, ...]} for accepted pairs.
    """
    method = params.get("method", "threshold")
    exclusive = params.get("exclusive", True)

    # Work on a minimal copy; keep src if present
    cols = ["s1_id", "cand_id", "p"]
    has_src = "src" in df.columns
    if has_src:
        cols = ["s1_id", "cand_id", "src", "p"]
    d = _assign_and_rank(df[cols].copy(), exclusive)

    if method == "threshold":
        return _decide_threshold(d, params, has_src)
    elif method == "expected_f05":
        return _decide_expected_f05(d, df, params)
    elif method == "prob_sum":
        return _decide_prob_sum(d, params)
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'threshold', 'expected_f05', or 'prob_sum'.")


# ------------------------------------------------------------------ threshold

def _decide_threshold(d, params, has_src):
    """Per-source, per-rank threshold selection."""
    t1_s2 = params["t1_s2"]
    t2_s2 = params["t2_s2"]
    t1_s3 = params.get("t1_s3", t1_s2)
    t2_s3 = params.get("t2_s3", t2_s2)
    # t3 overrides t2_sx for rank >= 3 (optional)
    t3 = params.get("t3", None)

    p = d["p"].values
    rank = d["rank"].values

    if has_src and "src" in d.columns:
        src = d["src"].values
        is_s3 = (src == 3) | (src == "3") | (src == "S3")
    else:
        # No src column: treat all as S2
        is_s3 = np.zeros(len(d), dtype=bool)

    t1 = np.where(is_s3, t1_s3, t1_s2)
    t2 = np.where(is_s3, t2_s3, t2_s2)

    # build effective threshold per row
    thr = np.where(rank == 1, t1, t2)
    if t3 is not None:
        thr = np.where(rank >= 3, t3, thr)

    keep = p >= thr
    return d[keep].groupby("s1_id")["cand_id"].apply(list).to_dict()


# ------------------------------------------------------------------ expected F0.5

def _f05_expected(tp_ex, pred, true_cnt):
    """Vectorised expected F0.5, treating TP as a real-valued expectation."""
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 1.25 * tp_ex / (1.25 * tp_ex + 0.25 * (true_cnt - tp_ex) + (pred - tp_ex))
    return np.where(true_cnt == 0, (pred == 0).astype(float),
                    np.where(tp_ex == 0, 0.0, f))


def _decide_expected_f05(d, df_orig, params):
    """Per-entity greedy top-k that maximises expected F0.5.

    For each entity we iterate k = 0..m (all candidates) and pick the k that
    maximises E[F0.5], where:
      E[TP | k] = sum of p for the top-k candidates
      E[n_true] comes from params or a 'n_true_found' column if present

    Retrieval misses (true matches not in the candidate list) are accounted for
    via n_true_est: E[TP] is clipped to n_true_est.
    """
    max_k = int(params.get("max_k", 30))
    # n_true estimate: per-entity column preferred, else fixed scalar
    n_true_default = float(params.get("n_true_est", 3.7))

    # try to get per-entity n_true from original df
    if "n_true_found" in df_orig.columns:
        nt_map = df_orig.groupby("s1_id")["n_true_found"].first().to_dict()
    else:
        nt_map = {}

    result = {}
    for s1_id, grp in d.groupby("s1_id"):
        probs = grp["p"].values          # already sorted desc by _assign_and_rank
        cands = grp["cand_id"].values
        n_true_est = float(nt_map.get(s1_id, n_true_default))
        m = min(len(probs), max_k)

        best_f, best_k = -1.0, 0
        cum_tp = 0.0
        for k in range(m + 1):
            if k > 0:
                cum_tp += probs[k - 1]
            # expected TP clipped to n_true_est (can't find more than exist)
            tp_ex = min(cum_tp, n_true_est)
            pred = float(k)
            if n_true_est == 0:
                f = 1.0 if k == 0 else 0.0
            elif tp_ex == 0:
                f = 0.0
            else:
                f = 1.25 * tp_ex / (1.25 * tp_ex + 0.25 * (n_true_est - tp_ex) + (pred - tp_ex))
            if f > best_f:
                best_f, best_k = f, k

        if best_k > 0:
            result[s1_id] = list(cands[:best_k])
    return result


# ------------------------------------------------------------------ probability sum

def _decide_prob_sum(d, params):
    """Accept round(sum(p)) candidates per entity, with rank-1 and minimum-p guards."""
    ps_t1 = float(params.get("ps_t1", 0.02))
    ps_tmin = float(params.get("ps_tmin", 0.10))
    ps_cap = int(params.get("ps_cap", 30))

    result = {}
    for s1_id, grp in d.groupby("s1_id"):
        probs = grp["p"].values          # sorted desc
        cands = grp["cand_id"].values
        rank = grp["rank"].values

        # rank-1 gate: if the best candidate is below ps_t1, skip entirely
        if len(probs) == 0 or probs[0] < ps_t1:
            continue

        # target k = round(sum(p)), capped
        k_target = int(round(float(probs.sum())))
        k_target = max(1, min(k_target, ps_cap, len(probs)))

        # apply minimum-p filter (can only reduce k, never below 1 if rank-1 passed)
        mask = probs >= ps_tmin
        mask[0] = True  # rank-1 already passed gate above
        # count how many of the top-k_target pass the minimum-p filter
        k_actual = int(mask[:k_target].sum())
        k_actual = max(1, k_actual)

        result[s1_id] = list(cands[:k_actual])
    return result


# ================================================================== L2: tune_v2
#
# Tunes decide_v2 params on the tune_half of businesses and evaluates on the
# held_half, to prevent overfitting to the tuning set.
#
# half_selector: a callable (s1_id -> bool) where True = tune half.
#                Default: hash(s1_id) % 2 == 0 (stable, deterministic split).

def _default_half(s1_id):
    return hash(s1_id) % 2 == 0


def tune_v2(df, truth, method="threshold", exclusive=True, half_selector=None,
            grid1=None, grid2=None, grid3=None):
    """Tune decide_v2 params on the tune half, return held-out F0.5 and best params.

    Parameters
    ----------
    df : DataFrame with columns s1_id, cand_id, src, p, label
    truth : dict {s1_id: set of true cand_ids} for all validation businesses
    method : "threshold" | "expected_f05" | "prob_sum"
    exclusive : bool
    half_selector : callable(s1_id) -> bool; True = tune half
    grid1, grid2, grid3 : np arrays of threshold values to sweep

    Returns
    -------
    (tune_f05, held_f05, best_params)
    """
    if half_selector is None:
        half_selector = _default_half

    all_s1 = list(truth.keys())
    tune_ids = set(s1 for s1 in all_s1 if half_selector(s1))
    held_ids = set(s1 for s1 in all_s1 if not half_selector(s1))

    df_tune = df[df["s1_id"].isin(tune_ids)]
    df_held = df[df["s1_id"].isin(held_ids)]
    truth_tune = {k: v for k, v in truth.items() if k in tune_ids}
    truth_held = {k: v for k, v in truth.items() if k in held_ids}

    if method == "threshold":
        best_params = _tune_threshold_v2(df_tune, truth_tune, exclusive, grid1, grid2, grid3)
    elif method == "expected_f05":
        best_params = _tune_expected_f05(df_tune, truth_tune, exclusive)
    elif method == "prob_sum":
        best_params = _tune_prob_sum(df_tune, truth_tune, exclusive, grid1, grid2)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    # evaluate on both halves
    tune_f05 = _score_v2(df_tune, truth_tune, best_params)
    held_f05 = _score_v2(df_held, truth_held, best_params)
    return tune_f05, held_f05, best_params


def _score_v2(df, truth, params):
    """Compute macro F0.5 using decide_v2(df, params) against truth."""
    if not truth:
        return float("nan")
    pred_map = decide_v2(df, params)
    total = 0.0
    for s1_id, true_set in truth.items():
        pred_set = set(pred_map.get(s1_id, []))
        true_set = set(true_set)
        if not true_set:
            total += 1.0 if not pred_set else 0.0
        else:
            tp = len(pred_set & true_set)
            if tp == 0:
                total += 0.0
            else:
                fp = len(pred_set) - tp
                fn = len(true_set) - tp
                total += 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
    return total / len(truth)


# ------------------------------------------------------------------ threshold tuner (v2)

def _tune_threshold_v2(df, truth, exclusive, grid1, grid2, grid3):
    """Grid-search per-source t1/t2 and optional t3 on the tune half."""
    grid1 = grid1 if grid1 is not None else np.round(np.arange(0.02, 0.96, 0.04), 3)
    grid2 = grid2 if grid2 is not None else np.round(np.arange(0.10, 0.99, 0.04), 3)
    # t3 grid: either None (disabled) or a coarse sweep
    grid3 = grid3 if grid3 is not None else np.concatenate([[999.0],  # 999 = disabled
                                                             np.round(np.arange(0.40, 0.99, 0.04), 3)])

    ents = list(truth.keys())
    code = {e: i for i, e in enumerate(ents)}
    n = len(ents)
    true_cnt = np.array([len(truth[e]) for e in ents], dtype=np.float64)

    has_src = "src" in df.columns
    cols = ["s1_id", "cand_id", "p", "label"]
    if has_src:
        cols = ["s1_id", "cand_id", "src", "p", "label"]

    d = _assign_and_rank(df[cols].copy(), exclusive)
    d = d[d["s1_id"].isin(code)].reset_index(drop=True)

    ent = d["s1_id"].map(code).values
    rank = d["rank"].values
    p = d["p"].values
    lab = d["label"].values.astype(bool)

    if has_src and "src" in d.columns:
        src_vals = d["src"].values
        is_s3 = (src_vals == 3) | (src_vals == "3") | (src_vals == "S3")
    else:
        is_s3 = np.zeros(len(d), dtype=bool)

    best = (-1.0, None)

    for t1_s2, t2_s2 in itertools.product(grid1, grid2):
        if t2_s2 < t1_s2:
            continue
        for t1_s3, t2_s3 in itertools.product(grid1, grid2):
            if t2_s3 < t1_s3:
                continue
            # Compute per-row effective threshold (without t3)
            t1_arr = np.where(is_s3, t1_s3, t1_s2)
            t2_arr = np.where(is_s3, t2_s3, t2_s2)
            base_thr = np.where(rank == 1, t1_arr, t2_arr)

            for t3_val in grid3:
                if t3_val >= 999.0:
                    thr = base_thr
                    t3_param = None
                else:
                    thr = np.where(rank >= 3, t3_val, base_thr)
                    t3_param = float(t3_val)

                keep = p >= thr
                pred_cnt = np.bincount(ent, weights=keep.astype(float), minlength=n)
                tp_cnt = np.bincount(ent, weights=(keep & lab).astype(float), minlength=n)
                with np.errstate(divide="ignore", invalid="ignore"):
                    f = 1.25 * tp_cnt / (1.25 * tp_cnt + 0.25 * (true_cnt - tp_cnt) + (pred_cnt - tp_cnt))
                f = np.where(true_cnt == 0, (pred_cnt == 0).astype(float),
                             np.where(tp_cnt == 0, 0.0, f))
                score = float(f.mean())

                if score > best[0]:
                    best = (score, {
                        "method": "threshold",
                        "exclusive": exclusive,
                        "t1_s2": float(t1_s2),
                        "t2_s2": float(t2_s2),
                        "t1_s3": float(t1_s3),
                        "t2_s3": float(t2_s3),
                        **({"t3": t3_param} if t3_param is not None else {}),
                    })
    return best[1]


# ------------------------------------------------------------------ expected-F0.5 tuner

def _tune_expected_f05(df, truth, exclusive):
    """Expected-F0.5 method has no free thresholds — just validate n_true_est."""
    # Try a few values of n_true_est
    best = (-1.0, None)
    cols = ["s1_id", "cand_id", "p"]
    if "src" in df.columns:
        cols = ["s1_id", "cand_id", "src", "p"]
    d = _assign_and_rank(df[cols].copy(), exclusive)
    for n_true_est in [2.5, 3.0, 3.7, 4.0, 4.5, 5.0]:
        params = {"method": "expected_f05", "exclusive": exclusive,
                  "n_true_est": n_true_est, "max_k": 30}
        f = _score_v2(d, truth, params)
        if f > best[0]:
            best = (f, params)
    return best[1]


# ------------------------------------------------------------------ prob-sum tuner

def _tune_prob_sum(df, truth, exclusive, grid1, grid2):
    """Grid-search ps_t1 and ps_tmin for prob_sum method."""
    grid1 = grid1 if grid1 is not None else np.round(np.arange(0.02, 0.40, 0.04), 3)
    grid2 = grid2 if grid2 is not None else np.round(np.arange(0.05, 0.60, 0.04), 3)
    cols = ["s1_id", "cand_id", "p"]
    if "src" in df.columns:
        cols = ["s1_id", "cand_id", "src", "p"]
    d = _assign_and_rank(df[cols].copy(), exclusive)
    best = (-1.0, None)
    for t1, tmin in itertools.product(grid1, grid2):
        params = {"method": "prob_sum", "exclusive": exclusive,
                  "ps_t1": float(t1), "ps_tmin": float(tmin), "ps_cap": 30}
        f = _score_v2(d, truth, params)
        if f > best[0]:
            best = (f, params)
    return best[1]


# ================================================================== convenience: full tune run

def run_full_tune(oof_path, truth_path, out_params_path, out_report_path,
                  methods=("threshold", "expected_f05", "prob_sum"),
                  half_selector=None):
    """Load run3_oof.parquet + run3_truth.parquet and tune all methods.

    Writes decision_params.json and the held-out F0.5 report.
    This function is called by the __main__ block below.

    Parameters
    ----------
    oof_path      : path to run3_oof.parquet (s1_id, cand_id, src, country, p, label)
    truth_path    : path to run3_truth.parquet (s1_id, country, n_true, n_true_found)
    out_params_path : path to write decision_params.json
    out_report_path : path to write the plain-text report
    methods       : which methods to tune
    half_selector : callable(s1_id) -> bool  (True = tune half)
    """
    import json
    import pandas as pd
    from .scoring import macro_f05

    print("Loading OOF data …")
    oof = pd.read_parquet(oof_path)
    truth_df = pd.read_parquet(truth_path)

    # Build truth dict: {s1_id: set_of_true_cand_ids}
    # truth_df gives n_true/n_true_found per entity; for scoring we need the actual cand_ids.
    # The label column in oof identifies true pairs, so we reconstruct from there.
    truth_map = (oof[oof["label"] == 1]
                 .groupby("s1_id")["cand_id"]
                 .apply(set)
                 .to_dict())
    # Include entities with 0 true matches (singletons): must appear in truth_map as empty set
    for s1_id in oof["s1_id"].unique():
        if s1_id not in truth_map:
            truth_map[s1_id] = set()
    # Also add entities from truth_df that have no candidates at all (blocking misses)
    # These need to be included so their 0-prediction is scored correctly
    for s1_id in truth_df["s1_id"].unique():
        if s1_id not in truth_map:
            # n_true > 0 means they have true matches but none were retrieved
            row = truth_df[truth_df["s1_id"] == s1_id].iloc[0]
            if row["n_true"] > 0:
                # We don't know the actual cand_ids, but n_true > 0 means empty pred = 0
                # Use a dummy set of the right size to correctly penalise misses
                truth_map[s1_id] = {f"__miss_{s1_id}_{i}" for i in range(int(row["n_true"]))}
            else:
                truth_map[s1_id] = set()

    # Baseline: v1 tune
    print("Computing baseline (v1) …")
    best_f_v1, t1_v1, t2_v1 = tune_thresholds(oof, truth_map)
    print(f"  Baseline v1: F0.5={best_f_v1:.4f}  t1={t1_v1}  t2={t2_v1}")

    # Tune and evaluate each method
    results = []
    best_overall = (-1.0, None, None)

    for method in methods:
        print(f"Tuning method={method!r} …")
        tune_f, held_f, params = tune_v2(oof, truth_map, method=method,
                                          half_selector=half_selector)
        results.append({
            "method": method,
            "tune_f05": round(tune_f, 6),
            "held_f05": round(held_f, 6),
            "params": params,
        })
        print(f"  {method}: tune={tune_f:.4f}  held={held_f:.4f}")
        if held_f > best_overall[0]:
            best_overall = (held_f, method, params)

    # Write params JSON
    _, best_method, best_params = best_overall
    output = {
        "selected_method": best_method,
        "baseline_v1": {"t1": t1_v1, "t2": t2_v1, "oof_f05": round(best_f_v1, 6)},
        "params": best_params,
        "all_methods": results,
    }
    import json
    with open(out_params_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_params_path}")

    # Write human-readable report
    lines = [
        "# L2 Decision Tuning Report",
        "",
        f"OOF file : {oof_path}",
        f"Truth file: {truth_path}",
        "",
        "## Baseline (v1)",
        f"  t1={t1_v1}  t2={t2_v1}  OOF macro F0.5 = {best_f_v1:.6f}",
        "",
        "## Per-method held-out F0.5 (tuned on first half, evaluated on second half)",
        "",
        f"{'Method':<18} {'Tune F0.5':>12} {'Held F0.5':>12}",
        "-" * 44,
    ]
    for r in results:
        lines.append(f"{r['method']:<18} {r['tune_f05']:>12.6f} {r['held_f05']:>12.6f}")
    lines += [
        "",
        f"Selected method: {best_method}  (best held-out F0.5 = {best_overall[0]:.6f})",
        "",
        "## Selected params",
        json.dumps(best_params, indent=2),
        "",
        "## Interpretation",
        "  Tune half = hash(s1_id) % 2 == 0 (deterministic, no leakage).",
        "  Held-out F0.5 is the honest estimate of leaderboard gain.",
        "  Params written to decision_params.json for use by L1 in run 5.",
    ]
    report = "\n".join(lines)
    with open(out_report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Wrote {out_report_path}")
    print(report)
    return output


# ================================================================== CLI entrypoint

if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description="L2: tune decision layer on run3/run4 OOF data")
    ap.add_argument("--oof", required=True, help="Path to run3_oof.parquet")
    ap.add_argument("--truth", required=True, help="Path to run3_truth.parquet")
    ap.add_argument("--params-out", default="decision_params.json",
                    help="Output path for decision_params.json")
    ap.add_argument("--report-out", default="l2_report.txt",
                    help="Output path for the held-out F0.5 report")
    ap.add_argument("--methods", default="threshold,expected_f05,prob_sum",
                    help="Comma-separated list of methods to tune")
    a = ap.parse_args()

    run_full_tune(
        oof_path=a.oof,
        truth_path=a.truth,
        out_params_path=a.params_out,
        out_report_path=a.report_out,
        methods=tuple(m.strip() for m in a.methods.split(",")),
    )
