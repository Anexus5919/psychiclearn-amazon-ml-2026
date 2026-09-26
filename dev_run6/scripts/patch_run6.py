"""Run-6 patch: dense-retrieval candidates as an extra blocking pass + 2 features (rank_dense, cos_dense).
Usage: patch_run6.py <src dir containing ber/>   (apply to a copy first; to the live tree only after run 5)"""
import io, os, py_compile, sys

B = os.path.join(sys.argv[1], "ber")


def patch(name, rep):
    p = os.path.join(B, name)
    s = io.open(p, encoding="utf-8").read()
    for a, b in rep:
        assert s.count(a) == 1, (name, a[:80])
        s = s.replace(a, b)
    io.open(p + ".new", "w", encoding="utf-8", newline="\n").write(s)
    py_compile.compile(p + ".new", doraise=True)
    return p


done = [patch("blocking.py", [
('''REGION_PASS = len(PASSES) + 1  # pass id of the region-restricted name pass (the reverse pass is len(PASSES))
''',
'''REGION_PASS = len(PASSES) + 1  # pass id of the region-restricted name pass (the reverse pass is len(PASSES))
DENSE_PASS = len(PASSES) + 2   # pass id of the dense-retrieval pass (fine-tuned e5 bi-encoder neighbours)
'''),
('''def block_partition(s1, pool, query_mask, cfg, log=print, tmp_dir=None):''',
'''def block_partition(s1, pool, query_mask, cfg, log=print, tmp_dir=None, dense=None):'''),
('''    all_keys = np.concatenate(keys)
''',
'''    if dense is not None and len(dense):
        # dense pass: precomputed nearest neighbours (s1 / pool positions, dense_cos, dense_rank)
        dkeys = dense["s1"].values.astype(np.int64) * n_pool + dense["pool"].values.astype(np.int64)
        keys.append(dkeys)
        passes.append(np.full(len(dense), DENSE_PASS, np.int8))
        ranks.append(dense["dense_rank"].values.astype(np.int16))
        log(f"    pass dense: {len(dense):,} pairs")
    all_keys = np.concatenate(keys)
'''),
('''    ks["rev"], ks["region"] = cfg.k_reverse, getattr(cfg, "k_region", 0)
    for pi, kind in enumerate(list(PASSES) + ["rev", "region"]):''',
'''    ks["rev"], ks["region"], ks["dense"] = cfg.k_reverse, getattr(cfg, "k_region", 0), getattr(cfg, "k_dense", 15)
    for pi, kind in enumerate(list(PASSES) + ["rev", "region", "dense"]):'''),
('''    if "region" in s1.columns and "region" in pool.columns:  # +1 same region, -1 different, 0 unknown''',
'''    if dense is not None and len(dense):
        dcos = pd.Series(dense["dense_cos"].values.astype(np.float32), index=dkeys)
        dcos = dcos[~dcos.index.duplicated()]
        cand["cos_dense"] = dcos.reindex(uniq).fillna(0.0).values.astype(np.float32)
    else:
        cand["cos_dense"] = np.float32(0.0)
    if "region" in s1.columns and "region" in pool.columns:  # +1 same region, -1 different, 0 unknown'''),
])]
done.append(patch("features.py", [
('''    "rank_region", "region_match",
    # competition''',
'''    "rank_region", "region_match", "rank_dense", "cos_dense",
    # competition'''),
('''                "rank_region", "region_match", "rev_is_best",''',
'''                "rank_region", "region_match", "rank_dense", "cos_dense", "rev_is_best",'''),
]))
done.append(patch("prune.py", [
('''CHEAP = ["cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "rank_region", "region_match",''',
'''CHEAP = ["cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "rank_region", "region_match",
         "rank_dense", "cos_dense",'''),
]))
done.append(patch("pipeline.py", [
('''    k_region: int = 0               # >0 enables the region-restricted name pass (see ber.regions)
''',
'''    k_region: int = 0               # >0 enables the region-restricted name pass (see ber.regions)
    dense_dir: str = ""             # dir with dense_{train,test}.parquet (fine-tuned e5 bi-encoder neighbours)
    k_dense: int = 15               # neighbours per S1 in those files (rank_dense = k_dense + 1 when absent)
'''),
('''        tmp_dir = io_utils.ensure_dir(os.path.join(cfg.work_dir, "tmp_block"))
''',
'''        tmp_dir = io_utils.ensure_dir(os.path.join(cfg.work_dir, "tmp_block"))
        def dense_for(country):  # per-country file if present (keeps RAM low), else the full file
            fp = os.path.join(cfg.dense_dir, f"dense_{split}_{country}.parquet")
            d = pd.read_parquet(fp if os.path.exists(fp) else os.path.join(cfg.dense_dir, f"dense_{split}.parquet"),
                                columns=["s1_id", "cand_id", "dense_cos", "dense_rank"])
            return d[d["dense_rank"] <= cfg.k_dense]
'''),
('''                cand = blocking.block_partition(s1, pool, qmask, cfg, log=lambda m: log(m, cfg), tmp_dir=tmp_dir)''',
'''                dense = None
                if cfg.dense_dir:
                    d = dense_for(country)
                    d = d[d["cand_id"].str.startswith(f"S{src}-")]
                    si = pd.Index(s1["entity_id"]).get_indexer(d["s1_id"])
                    pi = pd.Index(pool["entity_id"]).get_indexer(d["cand_id"])
                    ok = (si >= 0) & (pi >= 0)
                    ok &= qmask[np.where(si >= 0, si, 0)]
                    dense = pd.DataFrame({"s1": si[ok], "pool": pi[ok], "dense_cos": d["dense_cos"].values[ok],
                                          "dense_rank": d["dense_rank"].values[ok]})
                cand = blocking.block_partition(s1, pool, qmask, cfg, log=lambda m: log(m, cfg), tmp_dir=tmp_dir, dense=dense)'''),
('''                      ("k_reverse", int), ("exact_name_cap", int), ("k_region", int), ("cand_min_p", float),''',
'''                      ("k_reverse", int), ("exact_name_cap", int), ("k_region", int), ("cand_min_p", float),
                      ("dense_dir", str), ("k_dense", int),'''),
('''    for k in ("n_jobs", "train_frac", "k_name", "k_addr", "k_combo", "k_reverse", "exact_name_cap", "k_region", "cand_min_p",''',
'''    for k in ("n_jobs", "train_frac", "k_name", "k_addr", "k_combo", "k_reverse", "exact_name_cap", "k_region", "cand_min_p",
              "dense_dir", "k_dense",'''),
]))
for p in done:
    os.replace(p + ".new", p)
print("patched", [os.path.basename(p) for p in done], "in", B)
