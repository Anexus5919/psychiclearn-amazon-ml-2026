"""Pairwise features for (Source-1 record, candidate record) pairs.

Groups (see blueprint L4): name string similarities, address similarities, house-number
agreement / conflict, specificity (how common a name / address is), blocking scores and ranks,
competition context (does another S1 record claim this candidate more strongly?) and record
flags. Country is deliberately NOT a feature so the model transfers to the unseen country.
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

FEATURES = [
    # name
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "n_nospace_ratio", "n_nospace_partial",
    "n_acr_match", "legal_equal", "legal_one_missing", "p_native", "p_handle", "p_dba",
    "s1_ntok", "p_ntok",
    # address
    "a_tset", "a_tsort", "a_partial", "a_ratio", "p_addr_empty", "s1_ntok_addr", "p_ntok_addr",
    # numbers
    "num_jacc", "num_primary_eq", "num_any_overlap", "num_conflict", "num_primary_lev", "num_both",
    # blocking scores / ranks
    "cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "rank_rev",
    # competition / context
    "rev_is_best", "rev_best_score", "rev_second_score", "rev_gap",
    "ctx_rank_combo_src", "ctx_rank_combo_all", "ctx_gap_combo", "ctx_ncand", "ctx_rank_name_all",
    "ctx_rank_addr_all",
    # specificity
    "freq_s1_name", "freq_pool_name", "freq_s1_addr",
    # meta
    "src_is_s3",
]


def _cp(a, b, scorer, n_jobs, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=n_jobs, dtype=np.float32, **kw)


def _num_feats_chunk(args):
    a_list, b_list = args
    out = np.zeros((len(a_list), 4), dtype=np.float32)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        if not a or not b:
            continue
        sa, sb = set(a.split()), set(b.split())
        inter = len(sa & sb)
        out[i, 0] = inter / len(sa | sb)
        out[i, 1] = inter > 0
        out[i, 2] = inter == 0
        out[i, 3] = 1.0
    return out


NEEDED = ["name_core", "name_nospace", "name_acr", "legal", "name_native", "name_handle", "name_dba", "name_ntok",
          "addr_core", "addr_nums", "addr_primary", "addr_empty", "addr_ntok"]


def pair_features(cand, s1, pool, src, n_jobs, chunk=2_000_000):
    """Feature matrix for all candidate pairs of one partition (small inputs / tests)."""
    parts = [f for _, f in iter_pair_features(cand, s1, pool, src, n_jobs, chunk)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=FEATURES, dtype=np.float32)


def iter_pair_features(cand, s1, pool, src, n_jobs, chunk=1_000_000):
    """Yield (start_row, feature_chunk) for the candidate pairs of one partition, so callers can
    stream them to disk (the per-pair string columns exist for one chunk at a time only)."""
    s1_ctx = pd.DataFrame({
        "freq_s1_name": s1["name_core"].map(s1["name_core"].value_counts()).values,
        "freq_s1_addr": s1["addr_core"].map(s1["addr_core"].value_counts()).values,
    })
    pool_freq = pool["name_core"].map(pool["name_core"].value_counts()).values
    g = cand[["s1", "cos_combo", "cos_name", "cos_addr"]]
    ctx = pd.DataFrame({
        "ctx_rank_combo_src": g.groupby("s1")["cos_combo"].rank(ascending=False, method="min").values,
        "ctx_gap_combo": (g.groupby("s1")["cos_combo"].transform("max") - g["cos_combo"]).values,
        "ctx_ncand": g.groupby("s1")["cos_combo"].transform("size").values,
        "ctx_rank_name_all": g.groupby("s1")["cos_name"].rank(ascending=False, method="min").values,
        "ctx_rank_addr_all": g.groupby("s1")["cos_addr"].rank(ascending=False, method="min").values,
    })
    s1n, pooln = s1[NEEDED], pool[NEEDED]
    for s in range(0, len(cand), chunk):
        c = cand.iloc[s:s + chunk]
        f = _pair_features_chunk(c, s1n, pooln, src, n_jobs)
        f["freq_s1_name"] = s1_ctx["freq_s1_name"].values[c["s1"].values]
        f["freq_s1_addr"] = s1_ctx["freq_s1_addr"].values[c["s1"].values]
        f["freq_pool_name"] = pool_freq[c["pool"].values]
        for col in ctx.columns:
            f[col] = ctx[col].values[s:s + chunk]
        f["ctx_rank_combo_all"] = f["ctx_rank_combo_src"]  # recomputed across S2+S3 when pairs are loaded
        yield s, f[FEATURES].astype(np.float32)


def _pair_features_chunk(cand, s1, pool, src, n_jobs):
    """String / number / blocking features for one chunk of candidate pairs."""
    a = s1.iloc[cand["s1"].values].reset_index(drop=True)
    b = pool.iloc[cand["pool"].values].reset_index(drop=True)
    f = pd.DataFrame(index=a.index)
    an, bn = a["name_core"].tolist(), b["name_core"].tolist()
    f["n_ratio"] = _cp(an, bn, fuzz.ratio, n_jobs)
    f["n_tset"] = _cp(an, bn, fuzz.token_set_ratio, n_jobs)
    f["n_tsort"] = _cp(an, bn, fuzz.token_sort_ratio, n_jobs)
    f["n_partial"] = _cp(an, bn, fuzz.partial_ratio, n_jobs)
    f["n_jw"] = _cp(an, bn, JaroWinkler.normalized_similarity, n_jobs)
    ans, bns = a["name_nospace"].tolist(), b["name_nospace"].tolist()
    f["n_nospace_ratio"] = _cp(ans, bns, fuzz.ratio, n_jobs)
    f["n_nospace_partial"] = _cp(ans, bns, fuzz.partial_ratio, n_jobs)
    blen, alen = b["name_nospace"].str.len().values, a["name_nospace"].str.len().values
    f["n_acr_match"] = (((a["name_acr"].values == b["name_nospace"].values) & (blen >= 2) & (blen <= 6))
                        | ((b["name_acr"].values == a["name_nospace"].values) & (alen >= 2) & (alen <= 6))
                        ).astype(np.float32)
    f["legal_equal"] = (a["legal"].values == b["legal"].values).astype(np.float32)
    f["legal_one_missing"] = ((a["legal"].values == "") ^ (b["legal"].values == "")).astype(np.float32)
    f["p_native"] = b["name_native"].values.astype(np.float32)
    f["p_handle"] = b["name_handle"].values.astype(np.float32)
    f["p_dba"] = b["name_dba"].values.astype(np.float32)
    f["s1_ntok"] = a["name_ntok"].values.astype(np.float32)
    f["p_ntok"] = b["name_ntok"].values.astype(np.float32)

    aa, ba = a["addr_core"].tolist(), b["addr_core"].tolist()
    f["a_tset"] = _cp(aa, ba, fuzz.token_set_ratio, n_jobs)
    f["a_tsort"] = _cp(aa, ba, fuzz.token_sort_ratio, n_jobs)
    f["a_partial"] = _cp(aa, ba, fuzz.partial_ratio, n_jobs)
    f["a_ratio"] = _cp(aa, ba, fuzz.ratio, n_jobs)
    f["p_addr_empty"] = b["addr_empty"].values.astype(np.float32)
    f["s1_ntok_addr"] = a["addr_ntok"].values.astype(np.float32)
    f["p_ntok_addr"] = b["addr_ntok"].values.astype(np.float32)

    na, nb = a["addr_nums"].tolist(), b["addr_nums"].tolist()
    step = 200_000
    tasks = [(na[i:i + step], nb[i:i + step]) for i in range(0, len(na), step)]
    if n_jobs > 1 and len(tasks) > 1:
        with Pool(n_jobs) as pl:
            parts = pl.map(_num_feats_chunk, tasks)
    else:
        parts = [_num_feats_chunk(t) for t in tasks]
    nf = np.vstack(parts) if parts else np.zeros((0, 4), np.float32)
    f["num_jacc"], f["num_any_overlap"], f["num_conflict"], f["num_both"] = nf[:, 0], nf[:, 1], nf[:, 2], nf[:, 3]
    pa_, pb_ = a["addr_primary"].values, b["addr_primary"].values
    f["num_primary_eq"] = ((pa_ == pb_) & (pa_ != "")).astype(np.float32)
    lev = _cp(pa_.tolist(), pb_.tolist(), Levenshtein.distance, n_jobs)
    f["num_primary_lev"] = np.where((pa_ != "") & (pb_ != ""), lev, -1).astype(np.float32)

    for col in ["cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "rank_rev",
                "rev_is_best", "rev_best_score", "rev_second_score", "rev_gap"]:
        f[col] = cand[col].values.astype(np.float32)
    f["src_is_s3"] = np.float32(1.0 if src == 3 else 0.0)
    return f
