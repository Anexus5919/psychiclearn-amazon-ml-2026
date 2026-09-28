"""Candidate generation (blocking).

Within each (country, target source) partition we run three TF-IDF char-3-gram retrieval
passes and one reverse pass:

* name  : top-k by cosine on the normalised core name
* addr  : top-k by cosine on the normalised address (rescues invented / handle / acronym names)
* combo : top-k by cosine on "core name + address" (disambiguates generic names)
* reverse: for every pool record, its top-k Source-1 records by combo cosine (adds pairs and
  supplies "is this S1 the record's best S1?" competition context)

Country is a hard block: it agrees on 100% of matched training pairs. Every pair in the union is
re-scored exactly on all three representations. IDF is fitted on the unlabeled records being
matched (transductive; no labels are used).
"""
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # pragma: no cover - fallback keeps the code runnable without the package
    sp_matmul_topn = None

PASSES = ("name", "addr", "combo")
REGION_PASS = len(PASSES) + 1  # pass id of the region-restricted name pass (the reverse pass is len(PASSES))
DENSE_PASS = len(PASSES) + 2   # pass id of the dense-retrieval pass (fine-tuned e5 bi-encoder neighbours)

# retrieval-only name clean-up for synthetic noise seen in missed pairs: look-alike digits inside
# words ("internati0na1", "r0opesh") and injected record tags ("... (ID: 64721)")
_LEET = str.maketrans("013457", "oleast")
_LEET_TOK = re.compile(r"^(?=(?:.*[a-z]){2})(?=.*[013457])[a-z013457]+$")
_ID_TAG = re.compile(r"\bid \d+\b")


def _clean_name(n):
    n = _ID_TAG.sub(" ", n)
    return " ".join(t.translate(_LEET) if _LEET_TOK.match(t) else t for t in n.split())


def _texts(df, kind):
    if kind == "name":
        return [_clean_name(n) for n in df["name_core"].tolist()]
    if kind == "addr":
        return df["addr_core"].tolist()
    # name tokens get an "n:" prefix so they never collide with address tokens
    return [" ".join("n:" + t for t in _clean_name(n).split()) + " " + a
            for n, a in zip(df["name_core"].tolist(), df["addr_core"].tolist())]


def _vectorizer(kind, cfg):
    """Retrieval representation per pass (chosen by the speed/recall benchmark in the docs):
    names use char 3-grams (typo tolerant); address and combo passes use whole tokens, which were
    both more accurate and ~12x faster than char 3-grams on addresses."""
    common = dict(min_df=2, sublinear_tf=True, dtype=np.float32)
    if kind == "name":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), max_df=cfg.max_df_name, **common)
    max_df = cfg.max_df_addr if kind == "addr" else cfg.max_df_combo
    return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", max_df=max_df, **common)


_VEC = None


def _init_worker(vec):
    global _VEC
    _VEC = vec


def _transform_chunk(texts):
    return _VEC.transform(texts)


def _parallel_transform(vec, texts, n_jobs, chunk=100_000):
    from multiprocessing import Pool
    parts = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    if n_jobs > 1 and len(parts) > 1:
        with Pool(n_jobs, initializer=_init_worker, initargs=(vec,)) as pool:
            mats = pool.map(_transform_chunk, parts)
    else:
        mats = [vec.transform(p) for p in parts]
    return sparse.vstack(mats).tocsr() if mats else vec.transform([])


def _vectorise(s1_texts, pool_texts, kind, cfg, fit_sample=1_000_000, seed=0):
    """Fit TF-IDF (vocabulary + IDF from a random sample of both sides) and transform both sides
    in parallel; returns L2-normalised float32 CSR matrices with sorted indices."""
    vec = _vectorizer(kind, cfg)
    both = s1_texts + pool_texts
    if len(both) > fit_sample:
        rng = np.random.default_rng(seed)
        both = [both[i] for i in rng.choice(len(both), fit_sample, replace=False)]
    vec.fit(both)
    q = _parallel_transform(vec, s1_texts, cfg.n_jobs)
    p = _parallel_transform(vec, pool_texts, cfg.n_jobs)
    q.sort_indices()
    p.sort_indices()
    return q, p


def topk(a, b, k, n_threads):
    """Row-wise top-k of cosine(a, b) for L2-normalised CSR a (n x V) and b (m x V).

    Returns (idx, score) arrays of shape (n, k); missing slots have idx -1 and score 0.
    """
    n = a.shape[0]
    idx = np.full((n, k), -1, dtype=np.int64)
    sc = np.zeros((n, k), dtype=np.float32)
    if n == 0 or b.shape[0] == 0:
        return idx, sc
    if sp_matmul_topn is not None:
        c = sp_matmul_topn(a, b.T.tocsr(), top_n=k, threshold=1e-6, sort=True, n_threads=n_threads)
    else:
        rows = []
        bt = b.T.tocsr()
        for s in range(0, n, 2000):
            m = (a[s:s + 2000] @ bt).tocsr()
            for r in range(m.shape[0]):
                lo, hi = m.indptr[r], m.indptr[r + 1]
                order = np.argsort(-m.data[lo:hi])[:k]
                rows.append((m.indices[lo:hi][order], m.data[lo:hi][order]))
        indptr = np.cumsum([0] + [len(r[0]) for r in rows])
        c = sparse.csr_matrix((np.concatenate([r[1] for r in rows]) if rows else [],
                               np.concatenate([r[0] for r in rows]) if rows else [], indptr), shape=(n, b.shape[0]))
    c = c.tocsr()
    counts = np.diff(c.indptr)
    for r in np.nonzero(counts)[0]:
        lo, hi = c.indptr[r], c.indptr[r + 1]
        order = np.argsort(-c.data[lo:hi], kind="stable")[:k]
        idx[r, :len(order)] = c.indices[lo:hi][order]
        sc[r, :len(order)] = c.data[lo:hi][order]
    return idx, sc


def verify_topk(a, b, idx, sc, n_check=20, seed=0):
    """Independent brute-force check on a sample of rows: the returned scores must equal the exact
    cosines at the returned indices, and the k-th returned score must equal the exact k-th best.
    Raises AssertionError on any mismatch (a silent retrieval bug corrupts everything downstream).
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(a.shape[0], size=min(n_check, a.shape[0]), replace=False)
    bt = b.T.tocsr()
    k = idx.shape[1]
    for r in rows:
        exact = np.asarray((a[r] @ bt).todense()).ravel()
        got = idx[r][idx[r] >= 0]
        assert np.allclose(exact[got], sc[r][: len(got)], atol=1e-4), f"score mismatch at row {r}"
        kk = min(k, int((exact > 1e-6).sum()))
        if kk:
            kth = np.sort(exact)[::-1][kk - 1]
            assert abs(kth - sc[r][kk - 1]) < 1e-4, f"k-th score mismatch at row {r}: {kth} vs {sc[r][kk - 1]}"


def rowdot(a, b):
    """Row-wise dot products of two CSR matrices with equal shapes."""
    return np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)


def block_partition(s1, pool, query_mask, cfg, log=print, tmp_dir=None, dense=None):
    """Generate candidate pairs for one (country, source) partition.

    s1, pool   : normalised dataframes of this partition (index = position within the frame)
    query_mask : boolean array over s1 rows that need candidates (all rows for test; a sampled
                 subset for training). All s1 rows still act as competitors in the reverse pass.
    tmp_dir    : where TF-IDF matrices are spilled between passes (keeps only one pass in RAM).
    Returns a dataframe with one row per candidate pair and blocking-derived columns.
    """
    import os
    import tempfile
    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="ber_block_")
    ks = {"name": cfg.k_name, "addr": cfg.k_addr, "combo": cfg.k_combo}
    q_rows = np.nonzero(query_mask)[0]
    n_pool = np.int64(len(pool))
    keys, passes, ranks = [], [], []
    for pi, kind in enumerate(PASSES):
        q, p = _vectorise(_texts(s1, kind), _texts(pool, kind), kind, cfg)
        qa = q[q_rows]
        idx, sc = topk(qa, p, ks[kind], cfg.n_jobs)
        verify_topk(qa, p, idx, sc)
        rr, cc = np.nonzero(idx >= 0)
        keys.append(q_rows[rr].astype(np.int64) * n_pool + idx[rr, cc])
        passes.append(np.full(len(rr), pi, np.int8))
        ranks.append((cc + 1).astype(np.int16))
        log(f"    pass {kind:5s}: {len(rr):,} pairs")
        if kind == "name" and getattr(cfg, "k_region", 0) > 0 and "region" in s1.columns:
            # region-restricted name pass: same name TF-IDF, but each query only competes with pool
            # records of its own state/region (fixes country-wide "crowding" by same-named businesses)
            s1_reg, pool_reg = s1["region"].values, pool["region"].values
            pool_by = {r: g for r, g in pd.Series(np.arange(len(pool))).groupby(pool_reg)}
            q_reg = {r: g for r, g in pd.Series(q_rows).groupby(s1_reg[q_rows])}
            n_reg = 0
            for r, qs in q_reg.items():
                ps = pool_by.get(r)
                if not r or ps is None:
                    continue
                qr, pr = qs.values, ps.values
                ridx_, _ = topk(q[qr], p[pr], cfg.k_region, cfg.n_jobs)
                rr2, cc2 = np.nonzero(ridx_ >= 0)
                keys.append(qr[rr2].astype(np.int64) * n_pool + pr[ridx_[rr2, cc2]])
                passes.append(np.full(len(rr2), REGION_PASS, np.int8))
                ranks.append((cc2 + 1).astype(np.int16))
                n_reg += len(rr2)
            log(f"    pass region: {n_reg:,} pairs ({len(q_reg)} regions)")
        if kind == "combo" and cfg.k_reverse > 0:  # reverse pass: pool record -> its top S1 records
            ridx, rsc = topk(p, q, cfg.k_reverse, cfg.n_jobs)
            verify_topk(p, q, ridx, rsc)
            rr, cc = np.nonzero(ridx >= 0)
            s_of = ridx[rr, cc]
            keep = query_mask[s_of]
            keys.append(s_of[keep].astype(np.int64) * n_pool + rr[keep])
            passes.append(np.full(int(keep.sum()), len(PASSES), np.int8))
            ranks.append((cc[keep] + 1).astype(np.int16))
            log(f"    pass rev  : {int(keep.sum()):,} pairs (queried S1 only)")
        sparse.save_npz(os.path.join(tmp_dir, f"{kind}_q.npz"), q, compressed=False)
        sparse.save_npz(os.path.join(tmp_dir, f"{kind}_p.npz"), p, compressed=False)
        del q, p, qa

    if getattr(cfg, "exact_name_cap", 0) > 0:
        # exact core-name key: rescues candidates whose address is empty or garbled (the error
        # analysis showed exact-name records with empty addresses slipping past the TF-IDF passes);
        # names shared by more than `exact_name_cap` pool records are skipped (generic names)
        qn = pd.DataFrame({"s1": q_rows, "key": s1["name_core"].values[q_rows]})
        pn = pd.DataFrame({"pool": np.arange(len(pool)), "key": pool["name_core"].values})
        pn = pn[pn["key"].str.len() > 0]
        pn = pn[pn["key"].map(pn["key"].value_counts()) <= cfg.exact_name_cap]
        m = qn.merge(pn, on="key")
        keys.append(m["s1"].values.astype(np.int64) * n_pool + m["pool"].values.astype(np.int64))
        passes.append(np.full(len(m), 99, np.int8))  # adds pairs only; no rank column
        ranks.append(np.ones(len(m), np.int16))
        log(f"    pass exact: {len(m):,} pairs")

    if dense is not None and len(dense):
        # dense pass: precomputed nearest neighbours (s1 / pool positions, dense_cos, dense_rank)
        dkeys = dense["s1"].values.astype(np.int64) * n_pool + dense["pool"].values.astype(np.int64)
        keys.append(dkeys)
        passes.append(np.full(len(dense), DENSE_PASS, np.int8))
        ranks.append(dense["dense_rank"].values.astype(np.int16))
        log(f"    pass dense: {len(dense):,} pairs")
    all_keys = np.concatenate(keys)
    uniq, inv = np.unique(all_keys, return_inverse=True)
    all_pass, all_rank = np.concatenate(passes), np.concatenate(ranks)
    del keys, passes, ranks, all_keys
    cand = pd.DataFrame({"s1": (uniq // n_pool).astype(np.int64), "pool": (uniq % n_pool).astype(np.int64)})
    ks["rev"], ks["region"], ks["dense"] = cfg.k_reverse, getattr(cfg, "k_region", 0), getattr(cfg, "k_dense", 15)
    for pi, kind in enumerate(list(PASSES) + ["rev", "region", "dense"]):
        k = ks[kind]
        col = np.full(len(uniq), k + 1, np.int16)
        m = all_pass == pi
        col[inv[m]] = all_rank[m]
        cand[f"rank_{kind}"] = col
    s_idx, p_idx = cand["s1"].values, cand["pool"].values
    if dense is not None and len(dense):
        dcos = pd.Series(dense["dense_cos"].values.astype(np.float32), index=dkeys)
        dcos = dcos[~dcos.index.duplicated()]
        cand["cos_dense"] = dcos.reindex(uniq).fillna(0.0).values.astype(np.float32)
    else:
        cand["cos_dense"] = np.float32(0.0)
    if "region" in s1.columns and "region" in pool.columns:  # +1 same region, -1 different, 0 unknown
        ra, rb = s1["region"].values[s_idx], pool["region"].values[p_idx]
        known = (ra != "") & (rb != "")
        cand["region_match"] = np.where(known, np.where(ra == rb, 1, -1), 0).astype(np.int8)
    else:
        cand["region_match"] = np.int8(0)
    for kind in PASSES:  # exact cosines on every representation, one pass in memory at a time
        q = sparse.load_npz(os.path.join(tmp_dir, f"{kind}_q.npz")).tocsr()
        p = sparse.load_npz(os.path.join(tmp_dir, f"{kind}_p.npz")).tocsr()
        out = np.empty(len(cand), np.float32)
        for s in range(0, len(cand), 2_000_000):
            out[s:s + 2_000_000] = rowdot(q[s_idx[s:s + 2_000_000]], p[p_idx[s:s + 2_000_000]])
        cand[f"cos_{kind}"] = out
        del q, p
        for side in ("q", "p"):
            os.remove(os.path.join(tmp_dir, f"{kind}_{side}.npz"))
    if cfg.k_reverse > 0:
        best_s1 = ridx[:, 0]
        cand["rev_is_best"] = (best_s1[p_idx] == s_idx).astype(np.int8)
        cand["rev_best_score"] = rsc[p_idx, 0]
        cand["rev_second_score"] = rsc[p_idx, 1] if rsc.shape[1] > 1 else np.float32(0.0)
        cand["rev_gap"] = cand["rev_best_score"] - cand["cos_combo"]
    else:  # reverse pass disabled: constant columns (ignored by the model)
        for c in ("rev_is_best", "rev_best_score", "rev_second_score", "rev_gap"):
            cand[c] = np.float32(0.0)
    return cand
