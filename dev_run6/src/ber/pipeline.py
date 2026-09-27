"""End-to-end pipeline: data -> normalise -> blocking -> features -> LightGBM -> decisions -> TSVs.

Usage (from code/business_entity_resolution/src):
    python -m ber.pipeline --data-dir <dataset dir> --work-dir <scratch dir> --out-dir <output dir>
Stages can be run one at a time with --stage {prepare,pairs,train,predict,all}; every stage reads
the previous stage's files from --work-dir, so a failed run can be resumed.
"""
import argparse
import glob
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import blocking, decide, features, io_utils, model, normalize, scoring, writer

NORM_COLS = ["entity_id", "country", "name_core", "name_nospace", "name_acr", "legal", "name_native",
             "name_handle", "name_dba", "name_ntok", "addr_core", "addr_nums", "addr_primary", "addr_empty",
             "addr_ntok"]


@dataclass
class Config:
    data_dir: str
    work_dir: str
    out_dir: str
    n_jobs: int = min(8, os.cpu_count() or 1)  # leave cores free so the machine stays usable
    train_frac: float = 0.08   # share of train S1 entities used to build training pairs
    k_name: int = 10
    k_addr: int = 10
    k_combo: int = 15
    k_reverse: int = 0         # reverse pass (pool -> S1) disabled by default: costly on a laptop
    max_df_name: float = 0.005  # retrieval ignores terms present in more than this share of records
    max_df_addr: float = 0.01
    max_df_combo: float = 0.005
    n_folds: int = 4
    n_predict_models: int = 1
    exact_name_cap: int = 0         # >0 enables the exact core-name retrieval pass
    k_region: int = 0               # >0 enables the region-restricted name pass (see ber.regions)
    dense_dir: str = ""             # dir with dense_{train,test}.parquet (fine-tuned e5 bi-encoder neighbours)
    k_dense: int = 15               # neighbours per S1 in those files (rank_dense = k_dense + 1 when absent)
    cand_min_p: float = 0.0001      # candidate_pairs.tsv keeps pairs with matcher p >= this (run-4 OOF:
                                    # 22.6 -> 7.3 candidates per S1 for a 0.003% candidate-recall loss)
    only_countries: str = ""        # comma list: restrict the pairs stage to these countries
    prune: bool = True              # learned pruning of candidates (smaller candidate_pairs.tsv)
    prune_max_recall_loss: float = 0.002
    seed: int = 42


T0 = time.time()


def log(msg, cfg=None):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    if cfg is not None:
        with open(os.path.join(cfg.work_dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------ stage 1: normalise
def _write_norm(cfg, split, src, path, dicts=(None, None)):
    import pyarrow as pa
    import pyarrow.parquet as pq
    df = io_utils.read_source(cfg.data_dir, split, src)
    log(f"prepare {split} s{src}: {len(df):,} rows, countries={sorted(df['country'].unique())}", cfg)
    writer_ = None
    for chunk in normalize.iter_normalised(df, cfg.n_jobs, dicts=dicts):  # streamed chunk by chunk
        tb = pa.Table.from_pandas(chunk[NORM_COLS], preserve_index=False)
        writer_ = writer_ or pq.ParquetWriter(path + ".tmp", tb.schema)
        writer_.write_table(tb)
    writer_.close()
    os.replace(path + ".tmp", path)


def learn_translit(cfg):
    """Learn Indic->Latin dictionaries (names, addresses) from training ground-truth pairs.

    Leak-free: pairs of the S1 entities sampled as training/validation queries are excluded, so
    the out-of-fold validation score is not inflated by the dictionary.
    """
    from . import translit
    s1 = io_utils.read_source(cfg.data_dir, "train", 1)
    queries = train_query_ids(cfg, s1[["entity_id", "country"]])
    s1 = s1.set_index("entity_id")
    gt = io_utils.read_tsv(os.path.join(cfg.data_dir, "train", "train_ground_truth.tsv"),
                           ["source1_entity_id", "matched_entity_ids"])
    gt = gt[~gt["source1_entity_id"].isin(queries) & (gt["matched_entity_ids"] != "")]
    ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")[["source1_entity_id", "m"]]
    del gt
    name_pairs, addr_pairs = [], []
    for src in (2, 3):
        pool = io_utils.read_source(cfg.data_dir, "train", src)
        indic = r"[ऀ-෿]"  # Devanagari ... Sinhala blocks (all Indic scripts in the data)
        nat = pool[pool["business_name"].str.contains(indic, regex=True)
                   | pool["business_address"].str.contains(indic, regex=True)]
        del pool
        j = nat.merge(ex, left_on="entity_id", right_on="m", how="inner")
        name_pairs += list(zip(s1.loc[j["source1_entity_id"], "business_name"].values, j["business_name"].values))
        addr_pairs += list(zip(s1.loc[j["source1_entity_id"], "business_address"].values, j["business_address"].values))
    dn, st_n = translit.learn(name_pairs)
    da, st_a = translit.learn(addr_pairs)
    log(f"translit learned (excluding query entities): names {st_n}, addresses {st_a}", cfg)
    return dn, da


def stage_prepare(cfg):
    """Normalise every source file once (S2/S3 with learned transliteration) and store as Parquet."""
    out = io_utils.ensure_dir(os.path.join(cfg.work_dir, "norm"))
    for split in ("train", "test"):
        path = os.path.join(out, f"{split}_s1.parquet")
        if not os.path.exists(path):
            _write_norm(cfg, split, 1, path)
    tpath = os.path.join(out, "translit.json")
    if os.path.exists(tpath):
        d = json.load(open(tpath, encoding="utf-8"))
        dicts = (d["name"], d["addr"])
    else:
        dicts = learn_translit(cfg)
        json.dump({"name": dicts[0], "addr": dicts[1]}, open(tpath, "w", encoding="utf-8"), ensure_ascii=False)
    for split in ("train", "test"):
        for src in (2, 3):
            path = os.path.join(out, f"{split}_s{src}.parquet")
            if not os.path.exists(path):
                _write_norm(cfg, split, src, path, dicts)
    gt_path = os.path.join(out, "train_gt.parquet")
    if not os.path.exists(gt_path):
        gt = io_utils.read_ground_truth(cfg.data_dir)
        pd.DataFrame({"s1_id": list(gt.keys()), "matched": [",".join(sorted(v)) for v in gt.values()]}).to_parquet(gt_path, index=False)
    log("prepare done", cfg)


def stage_regions(cfg):
    """State/region key per record (train+test, S1/S2/S3) for the region-restricted name pass."""
    from . import regions
    out = os.path.join(cfg.work_dir, "norm")
    paths = {(sp, src): os.path.join(out, f"{sp}_s{src}_region.parquet") for sp in ("train", "test") for src in (1, 2, 3)}
    if all(os.path.exists(p) for p in paths.values()):
        return
    dic = json.load(open(os.path.join(out, "translit.json"), encoding="utf-8"))["addr"]
    cols = ["entity_id", "business_address", "country"]
    s1 = pd.concat([io_utils.read_source(cfg.data_dir, sp, 1)[cols] for sp in ("train", "test")], ignore_index=True)
    vocab, cmap = regions.learn_s1(s1)
    log(f"regions: vocabulary {({c: len(v) for c, v in vocab.items()})}, S1 component map {({c: len(m) for c, m in cmap.items()})}", cfg)
    qpath = os.path.join(cfg.work_dir, "pairs", "train", "_queries.parquet")
    queries = set(pd.read_parquet(qpath)["s1_id"]) if os.path.exists(qpath) else train_query_ids(cfg, s1)
    gt = io_utils.read_tsv(os.path.join(cfg.data_dir, "train", "train_ground_truth.tsv"),
                           ["source1_entity_id", "matched_entity_ids"])
    gt = gt[gt["matched_entity_ids"] != ""]
    ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")[["source1_entity_id", "m"]]
    del gt
    s1a = s1.set_index("entity_id")["business_address"]
    learn, check = [], []
    for src in (2, 3):
        pool = io_utils.read_source(cfg.data_dir, "train", src)[cols]
        j = ex.merge(pool, left_on="m", right_on="entity_id")
        del pool
        held = j["source1_entity_id"].isin(queries).values
        for part, dst, n in ((j[~held], learn, 400_000), (j[held], check, 50_000)):
            part = part.sample(min(n, len(part)), random_state=0)
            dst += list(zip(part["country"].values, s1a.loc[part["source1_entity_id"]].values,
                            part["business_address"].values))
    gmap = regions.learn_pairs([(c, a, regions.components(b, dic)) for c, a, b in learn], vocab, cmap)
    log(f"regions: learned pool-component map (query entities excluded) {({c: len(m) for c, m in gmap.items()})}", cfg)
    agree = Counter()
    for c, a, b in check:  # held-out check on the query entities' true pairs
        ra = regions.region_of(regions.components(a), vocab[c], cmap[c])
        rb = regions.region_of(regions.components(b, dic), vocab[c], cmap[c], gmap.get(c, {}))
        agree[(c, "both" if ra and rb else "missing", ra == rb if ra and rb else None)] += 1
    log(f"regions: held-out true pairs (country, coverage, same region): {dict(agree)}", cfg)
    json.dump({"vocab": {c: sorted(v) for c, v in vocab.items()}, "s1_map": cmap, "pair_map": gmap},
              open(os.path.join(out, "regions.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for (sp, src), p in paths.items():
        if os.path.exists(p):
            continue
        df = io_utils.read_source(cfg.data_dir, sp, src)[cols]
        r = regions.assign(df, vocab, cmap, gmap, dic if src != 1 else None, cfg.n_jobs)
        cov = (r["region"] != "").groupby(df["country"].values).mean().round(3).to_dict()
        r.to_parquet(p, index=False)
        log(f"regions {sp} s{src}: coverage {cov}", cfg)


def load_norm(cfg, split, src, country=None, columns=None):
    """Load normalised records, optionally only one country and a subset of columns."""
    filters = [("country", "==", country)] if country is not None else None
    return pd.read_parquet(os.path.join(cfg.work_dir, "norm", f"{split}_s{src}.parquet"),
                           columns=columns, filters=filters)


def load_gt(cfg):
    df = pd.read_parquet(os.path.join(cfg.work_dir, "norm", "train_gt.parquet"))
    return {s: set(m.split(",")) if m else set() for s, m in zip(df["s1_id"], df["matched"])}


def train_query_ids(cfg, s1):
    """Sample train S1 entities (stratified by country) whose candidates become training pairs."""
    rng = np.random.default_rng(cfg.seed)
    ids = []
    for c, g in s1.groupby("country"):
        n = max(1, int(round(len(g) * cfg.train_frac)))
        ids.extend(g["entity_id"].values[rng.choice(len(g), n, replace=False)])
    return set(ids)


# ------------------------------------------------------------------ stage 2: blocking + features
def stage_pairs(cfg, splits=("train", "test")):
    """Generate candidates and features for every (split, country, source) partition."""
    for split in splits:
        out = io_utils.ensure_dir(os.path.join(cfg.work_dir, "pairs", split))
        import pyarrow as pa
        import pyarrow.parquet as pq
        s1_meta = load_norm(cfg, split, 1, columns=["entity_id", "country"])
        qpath = os.path.join(out, "_queries.parquet")
        if os.path.exists(qpath):
            queries = set(pd.read_parquet(qpath)["s1_id"])
        else:
            queries = train_query_ids(cfg, s1_meta) if split == "train" else set(s1_meta["entity_id"])
            pd.DataFrame({"s1_id": sorted(queries)}).to_parquet(qpath, index=False)
        countries = sorted(s1_meta["country"].unique())
        del s1_meta
        tmp_dir = io_utils.ensure_dir(os.path.join(cfg.work_dir, "tmp_block"))
        def dense_for(country):  # per-country file if present (keeps RAM low), else the full file
            fp = os.path.join(cfg.dense_dir, f"dense_{split}_{country}.parquet")
            d = pd.read_parquet(fp if os.path.exists(fp) else os.path.join(cfg.dense_dir, f"dense_{split}.parquet"),
                                columns=["s1_id", "cand_id", "dense_cos", "dense_rank"])
            k_lim = 20 if country == "France" else cfg.k_dense
            return d[d["dense_rank"] <= k_lim]
        only = {c for c in cfg.only_countries.split(",") if c}
        for country in countries:
            if only and country not in only:
                continue
            for src in (2, 3):
                path = os.path.join(out, f"{country}_s{src}.parquet")
                if os.path.exists(path):
                    continue
                s1 = load_norm(cfg, split, 1, country=country)
                pool = load_norm(cfg, split, src, country=country)
                if cfg.k_region > 0:
                    for df_, s_ in ((s1, 1), (pool, src)):
                        reg = pd.read_parquet(os.path.join(cfg.work_dir, "norm", f"{split}_s{s_}_region.parquet"))
                        df_["region"] = df_["entity_id"].map(reg.set_index("entity_id")["region"]).fillna("").values
                        del reg
                qmask = s1["entity_id"].isin(queries).values
                log(f"pairs {split} {country} s{src}: S1={len(s1):,} (queries {qmask.sum():,}) pool={len(pool):,}", cfg)
                dense = None
                if cfg.dense_dir:
                    d = dense_for(country)
                    d = d[d["cand_id"].str.startswith(f"S{src}-")]
                    si = pd.Index(s1["entity_id"]).get_indexer(d["s1_id"])
                    pi = pd.Index(pool["entity_id"]).get_indexer(d["cand_id"])
                    ok = (si >= 0) & (pi >= 0)
                    ok &= qmask[np.where(si >= 0, si, 0)]
                    dense = pd.DataFrame({"s1": si[ok], "pool": pi[ok], "dense_cos": d["dense_cos"].values[ok],
                                          "dense_rank": d["dense_rank"].values[ok]})
                cand = blocking.block_partition(s1, pool, qmask, cfg, log=lambda m: log(m, cfg), tmp_dir=tmp_dir, dense=dense)
                log(f"    union: {len(cand):,} pairs; computing features", cfg)
                s1_ids, pool_ids = s1["entity_id"].values, pool["entity_id"].values
                writer_ = None
                for start, f in features.iter_pair_features(cand, s1, pool, src, cfg.n_jobs):
                    c = cand.iloc[start:start + len(f)]
                    f.insert(0, "cand_id", pool_ids[c["pool"].values])
                    f.insert(0, "s1_id", s1_ids[c["s1"].values])
                    tb = pa.Table.from_pandas(f, preserve_index=False)
                    writer_ = writer_ or pq.ParquetWriter(path + ".tmp", tb.schema)
                    writer_.write_table(tb)
                if writer_ is not None:
                    writer_.close()
                    os.replace(path + ".tmp", path)
                log(f"  -> {len(cand):,} pairs written ({len(cand) / max(1, qmask.sum()):.1f} per queried S1)", cfg)
                del cand, s1, pool
    log("pairs done", cfg)


def pair_files(cfg, split, sub="pairs"):
    """{country: [pair files]}: raw candidates (sub='pairs', one file per country and source) or
    pruned candidates (sub='pairs_pruned', one file per country)."""
    out = {}
    pattern = "*_s[23].parquet" if sub == "pairs" else "[!_]*.parquet"
    for f in sorted(glob.glob(os.path.join(cfg.work_dir, sub, split, pattern))):
        b = os.path.basename(f)
        out.setdefault(b.rsplit("_s", 1)[0] if sub == "pairs" else b[:-len(".parquet")], []).append(f)
    return out


def iter_country_pairs(cfg, split, batch_rows=2_000_000, sub="pairs", feats=None, recompute_ctx=True, countries=None):
    """Yield (keys_df, X) chunks per country with cross-source context filled in.

    ctx_rank_combo_all ranks each candidate by combo cosine across S2 and S3 together, so it is
    computed from the (small) key columns of both source files of a country before streaming the
    feature rows. Keys use Arrow-backed strings to keep memory low.
    """
    import pyarrow.parquet as pq
    feats = feats or features.FEATURES
    fidx = feats.index("ctx_rank_combo_all") if "ctx_rank_combo_all" in feats else None
    for country, files in pair_files(cfg, split, sub).items():
        if countries is not None and country not in countries:
            continue
        keys = pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id", "cos_combo"], dtype_backend="pyarrow")
                          for f in files], ignore_index=True)
        ranks = (keys.groupby("s1_id")["cos_combo"].rank(ascending=False, method="min").to_numpy(np.float32)
                 if recompute_ctx and fidx is not None else None)
        off = 0
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_rows, columns=feats):
                X = np.column_stack([batch.column(c).to_numpy(zero_copy_only=False) for c in feats]).astype(np.float32)
                n = X.shape[0]
                if ranks is not None:
                    X[:, fidx] = ranks[off:off + n]
                yield country, keys.iloc[off:off + n][["s1_id", "cand_id"]].reset_index(drop=True), X
                off += n


# ------------------------------------------------------------------ stage 2b: learned pruning
def stage_prune(cfg):
    """Train the cheap pre-ranker (out-of-fold), choose the keep rule, write pruned candidates."""
    from . import prune
    gt = load_gt(cfg)
    queries = pd.read_parquet(os.path.join(cfg.work_dir, "pairs", "train", "_queries.parquet"))["s1_id"].tolist()
    pk, px = [], []
    for _, k, x in iter_country_pairs(cfg, "train", feats=prune.CHEAP):
        pk.append(k)
        px.append(x)
    keys = pd.concat(pk, ignore_index=True)
    X = np.vstack(px)
    del pk, px
    y = np.fromiter((c in gt[s] for s, c in zip(keys["s1_id"].tolist(), keys["cand_id"].tolist())),
                    dtype=np.int8, count=len(keys))
    folds = model.fold_ids(keys["s1_id"].astype(str).values, cfg.n_folds)
    oof, models = prune.train_oof(X, y, folds, cfg.n_jobs)
    del X
    rank = prune.within_entity_rank(pd.factorize(keys["s1_id"])[0], oof)
    rule = prune.choose_rule(oof, rank, y.astype(bool), len(queries), cfg.prune_max_recall_loss)
    mdir = io_utils.ensure_dir(os.path.join(cfg.work_dir, "prune"))
    for i, m in enumerate(models):
        m.save_model(os.path.join(mdir, f"pre{i}.txt"), num_iteration=m.best_iteration)
    json.dump(rule, open(os.path.join(mdir, "rule.json"), "w"), indent=2)
    log(f"pre-ranker rule: {rule} (retrieval gave {len(keys) / len(queries):.1f} candidates per queried S1)", cfg)
    del keys, y, rank
    _write_pruned(cfg, "train", rule, oof=oof)
    _write_pruned(cfg, "test", rule, models=models[:1])  # one fold model: 4x faster on ~93M test pairs
    log("prune done", cfg)


def _write_pruned(cfg, split, rule, models=None, oof=None):
    """Keep each entity's top candidates by pre-ranker probability and write all features plus
    pre_p / pre_rank for the kept pairs (one file per country). Train uses out-of-fold scores."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from . import prune
    out = io_utils.ensure_dir(os.path.join(cfg.work_dir, "pairs_pruned", split))
    g_off, kept, total = 0, 0, 0
    for country, files in pair_files(cfg, split).items():
        n_c = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        if oof is not None:
            p = oof[g_off:g_off + n_c]
            g_off += n_c
        else:
            p = np.concatenate([
                np.mean([m.predict(x, num_iteration=m.best_iteration, num_threads=cfg.n_jobs) for m in models], axis=0)
                for _, _, x in iter_country_pairs(cfg, split, feats=prune.CHEAP, countries=[country])]).astype(np.float32)
        s1 = pd.concat([pd.read_parquet(f, columns=["s1_id"]) for f in files], ignore_index=True)["s1_id"]
        rank = prune.within_entity_rank(pd.factorize(s1)[0], p)
        keep = (rank <= rule["top_n"]) & (p >= rule["min_p"])
        path = os.path.join(out, f"{country}.parquet")
        writer_, off = None, 0
        for _, k, X in iter_country_pairs(cfg, split, countries=[country]):
            n = len(k)
            m = keep[off:off + n]
            df = pd.DataFrame(X[m], columns=features.FEATURES)
            df.insert(0, "cand_id", np.asarray(k["cand_id"].astype(str).values)[m])
            df.insert(0, "s1_id", np.asarray(k["s1_id"].astype(str).values)[m])
            df["pre_p"] = p[off:off + n][m]
            df["pre_rank"] = rank[off:off + n][m].astype(np.float32)
            tb = pa.Table.from_pandas(df, preserve_index=False)
            writer_ = writer_ or pq.ParquetWriter(path + ".tmp", tb.schema)
            writer_.write_table(tb)
            off += n
        writer_.close()
        os.replace(path + ".tmp", path)
        kept += int(keep.sum())
        total += n_c
        log(f"  pruned {split} {country}: kept {int(keep.sum()):,} of {n_c:,} pairs", cfg)
    log(f"pruned {split}: {kept:,} of {total:,} pairs kept ({kept / max(1, total):.1%})", cfg)


CE_COLS = ["ce_p", "ce_rank", "ce_gap", "ce_n50"]
CE2_COLS = ["ce2_p", "ce2_rank", "ce2_gap", "ce2_n50"]  # second cross-encoder (mDeBERTa), optional


def _main_inputs(cfg):
    """(sub-directory, features) for the main matcher: pruned pairs + pre-ranker features, plus any
    run-4 extra / cross-encoder columns present in the pruned files."""
    if cfg.prune and os.path.isdir(os.path.join(cfg.work_dir, "pairs_pruned", "test")):
        import pyarrow.parquet as pq
        from . import features_extra as fx
        feats = features.FEATURES + ["pre_p", "pre_rank"]
        f0 = sorted(glob.glob(os.path.join(cfg.work_dir, "pairs_pruned", "train", "[!_]*.parquet")))[0]
        names = set(pq.ParquetFile(f0).schema_arrow.names)
        feats += [c for c in fx.EXTRA + CE_COLS + CE2_COLS if c in names]
        return "pairs_pruned", feats
    return "pairs", features.FEATURES


def stage_augment(cfg):
    """Append run-4 extra + group features to every pruned candidate file (idempotent)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from . import features_extra as fx
    cols = ["entity_id", "name_core", "legal", "addr_core", "addr_nums", "addr_primary"]
    for split in ("train", "test"):
        for country, files in pair_files(cfg, split, "pairs_pruned").items():
            path = files[0]
            pf = pq.ParquetFile(path)
            if "g_pre_max" in pf.schema_arrow.names:
                continue
            keys = pd.read_parquet(path, columns=["s1_id", "cand_id", "pre_p"])
            G, top = fx.group_features(keys["s1_id"].values, keys["pre_p"].values, keys["cand_id"].values)
            del keys
            s1n = load_norm(cfg, split, 1, country=country, columns=cols).set_index("entity_id")
            pooln = pd.concat([load_norm(cfg, split, 2, country=country, columns=cols),
                               load_norm(cfg, split, 3, country=country, columns=cols)]).set_index("entity_id")
            writer_, off = None, 0
            for batch in pf.iter_batches(batch_size=1_000_000):
                df = batch.to_pandas()
                n = len(df)
                P = fx.pair_features(df["s1_id"].values, df["cand_id"].values, top[off:off + n], s1n, pooln, cfg.n_jobs)
                for c in fx.GROUP:
                    df[c] = G[c].values[off:off + n]
                for c in fx.PAIR:
                    df[c] = P[c].values
                tb = pa.Table.from_pandas(df, preserve_index=False)
                writer_ = writer_ or pq.ParquetWriter(path + ".aug", tb.schema)
                writer_.write_table(tb)
                off += n
            writer_.close()
            del pf
            os.replace(path + ".aug", path)
            log(f"  augmented {split} {country}: {off:,} pairs", cfg)
    log("augment done", cfg)


def stage_augment_ce(cfg, ce_dir, prefix="ce"):
    """Merge cross-encoder scores (ce_dir/<split>.parquet: s1_id, cand_id, ce_p) into the pruned files;
    unscored pairs get NaN. Adds per-entity rank / gap-to-best / count>=0.5 of the CE score.
    prefix "ce" = first cross-encoder (e5), "ce2" = second one (mDeBERTa)."""
    cols = [f"{prefix}_p", f"{prefix}_rank", f"{prefix}_gap", f"{prefix}_n50"]
    for split in ("train", "test"):
        ce = pd.read_parquet(os.path.join(ce_dir, f"{split}.parquet"), columns=["s1_id", "cand_id", "ce_p"])
        ce = ce.rename(columns={"ce_p": cols[0]})
        for country, files in pair_files(cfg, split, "pairs_pruned").items():
            path = files[0]
            df = pd.read_parquet(path)
            df = df.drop(columns=[c for c in cols if c in df.columns])
            df = df.merge(ce, on=["s1_id", "cand_id"], how="left")
            g = df.groupby("s1_id")[cols[0]]
            df[cols[1]] = g.rank(ascending=False, method="min").astype(np.float32)
            df[cols[2]] = (g.transform("max") - df[cols[0]]).astype(np.float32)
            df[cols[3]] = (df[cols[0]] >= 0.5).astype(np.float32).groupby(df["s1_id"]).transform("sum")
            df.to_parquet(path + ".ce", index=False)
            del df
            os.replace(path + ".ce", path)
            log(f"  {prefix} scores merged: {split} {country}", cfg)


# ------------------------------------------------------------------ stage 3: train + validate
def stage_train(cfg):
    """Out-of-fold LightGBM, threshold tuning and an end-to-end validation report."""
    gt = load_gt(cfg)
    import pyarrow.parquet as pq
    sub, feats = _main_inputs(cfg)
    n_rows = sum(pq.ParquetFile(f).metadata.num_rows for fs in pair_files(cfg, "train", sub).values() for f in fs)
    # disk-backed feature matrix: avoids holding (and briefly doubling) several GB in RAM
    X = np.lib.format.open_memmap(os.path.join(cfg.work_dir, "train_X.npy"), mode="w+", dtype=np.float32,
                                  shape=(n_rows, len(feats)))
    parts, off = [], 0
    for _, k, x in iter_country_pairs(cfg, "train", sub=sub, feats=feats, recompute_ctx=(sub == "pairs")):
        X[off:off + len(x)] = x
        parts.append(k)
        off += len(x)
    X.flush()
    df = pd.concat(parts, ignore_index=True)
    del parts
    queries = pd.read_parquet(os.path.join(cfg.work_dir, "pairs", "train", "_queries.parquet"))["s1_id"].tolist()
    df["label"] = np.fromiter((c in gt[s] for s, c in zip(df["s1_id"].tolist(), df["cand_id"].tolist())),
                              dtype=np.int8, count=len(df))
    log(f"train pairs: {len(df):,}, positives {int(df['label'].sum()):,}, queried S1 {len(queries):,}", cfg)
    oof, models = model.train_oof(X, df["label"].values, df["s1_id"].astype(str).values, cfg.n_folds, cfg.n_jobs,
                                  log=lambda m: log(m, cfg))
    del X
    mdir = io_utils.ensure_dir(os.path.join(cfg.work_dir, "models"))
    for k, m in enumerate(models):
        m.save_model(os.path.join(mdir, f"fold{k}.txt"), num_iteration=m.best_iteration)
    df["p"] = oof
    cands = df.groupby("s1_id")["cand_id"].apply(set).to_dict()
    df = df[df["p"] >= 0.01].reset_index(drop=True)  # can never be accepted (min threshold 0.02)

    truth = {s: gt[s] for s in queries}
    best_f, t1, t2 = decide.tune_thresholds(df, truth)
    s1_country = load_norm(cfg, "train", 1).set_index("entity_id")["country"]
    pred = decide.decide(df, t1, t2)
    report = {"config": asdict(cfg), "t1": t1, "t2": t2, "oof_macro_f05": best_f, "per_country": {}}
    true_pairs = sum(len(v) for v in truth.values())
    found = sum(len(truth[s] & cands.get(s, set())) for s in truth)
    report["pair_recall_of_blocking"] = found / max(1, true_pairs)
    report["candidates_per_s1"] = sum(len(v) for v in cands.values()) / len(truth)
    for c in sorted(s1_country[queries].unique()):
        ids = [s for s in queries if s1_country[s] == c]
        tr = {s: truth[s] for s in ids}
        report["per_country"][c] = {
            "n_s1": len(ids),
            "macro_f05": scoring.macro_f05(pred, tr),
            "oracle_f05_given_candidates": scoring.macro_f05({s: truth[s] & cands.get(s, set()) for s in ids}, tr),
        }
    report["main_inputs"] = {"pairs": sub, "features": feats}
    imp = pd.Series(models[0].feature_importance("gain"), index=feats).sort_values(ascending=False)
    report["top_features_gain"] = {k: float(v) for k, v in imp.head(20).items()}
    with open(os.path.join(cfg.work_dir, "report_train.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"OOF macro F0.5 = {best_f:.5f} (t1={t1}, t2={t2}); blocking pair recall = {report['pair_recall_of_blocking']:.4f}", cfg)
    for c, r in report["per_country"].items():
        log(f"  {c}: F0.5={r['macro_f05']:.5f}  oracle={r['oracle_f05_given_candidates']:.5f}  n={r['n_s1']:,}", cfg)


# ------------------------------------------------------------------ stage 4: predict + write
def stage_predict(cfg):
    """Score test candidates, apply the tuned decision rule and write both submission files."""
    rep = json.load(open(os.path.join(cfg.work_dir, "report_train.json")))
    files = sorted(glob.glob(os.path.join(cfg.work_dir, "models", "fold*.txt")))[: cfg.n_predict_models]
    # one fold model by default: 4x faster on ~100M test pairs, and consistent with the out-of-fold
    # predictions (each also came from a single fold model) on which the thresholds were tuned
    models = [lgb.Booster(model_file=f) for f in files]
    log(f"predicting with {len(models)} model(s): {[os.path.basename(f) for f in files]}", cfg)
    sub, feats = _main_inputs(cfg)
    cands, scored, n_pairs, last = {}, [], 0, None
    for country, keys, X in iter_country_pairs(cfg, "test", sub=sub, feats=feats, recompute_ctx=(sub == "pairs")):
        if country != last:
            log(f"  predicting {country} ({n_pairs:,} pairs scored so far)", cfg)
            last = country
        keys["p"] = model.predict(models, X, n_jobs=cfg.n_jobs)
        n_pairs += len(keys)
        scored.append(keys[keys["p"] >= 0.01])  # lower p can never be accepted (min threshold 0.02)
        # candidates stored as comma-joined chunks per S1 (compact); joined again when written
        kc = keys[keys["p"] >= cfg.cand_min_p]
        for s1, joined in kc.groupby("s1_id", sort=False)["cand_id"].agg(",".join).items():
            cands.setdefault(s1, []).append(joined)
    log(f"test pairs scored: {n_pairs:,}", cfg)
    df = pd.concat(scored, ignore_index=True)
    df.to_parquet(os.path.join(cfg.work_dir, "test_pred.parquet"), index=False)  # p >= 0.01 (pseudo-labels, analysis)
    pred = decide.decide(df, rep["t1"], rep["t2"])
    s1_ids = io_utils.read_source(cfg.data_dir, "test", 1)["entity_id"].tolist()
    mpath = os.path.join(cfg.out_dir, "matching_results.tsv")
    cpath = os.path.join(cfg.out_dir, "candidate_pairs.tsv")
    writer.write_id_lists(cpath, "candidate_entity_ids", s1_ids, cands)
    writer.write_id_lists(mpath, "matched_entity_ids", s1_ids, pred)
    targets = set(load_norm(cfg, "test", 2)["entity_id"]) | set(load_norm(cfg, "test", 3)["entity_id"])
    stats = writer.check_outputs(mpath, cpath, s1_ids, targets)
    json.dump(stats, open(os.path.join(cfg.work_dir, "report_predict.json"), "w"), indent=2)
    log(f"outputs written and self-checked: {stats}", cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage", default="all", choices=["prepare", "regions", "pairs", "prune", "augment", "augment_ce", "train", "predict", "all"])
    ap.add_argument("--ce-dir", default=None, help="directory with cross-encoder scores (train.parquet, test.parquet)")
    ap.add_argument("--ce-prefix", default="ce", help="feature prefix for --stage augment_ce: ce (e5) or ce2 (mDeBERTa)")
    for name, typ in (("n_jobs", int), ("train_frac", float), ("k_name", int), ("k_addr", int), ("k_combo", int),
                      ("k_reverse", int), ("exact_name_cap", int), ("k_region", int), ("cand_min_p", float),
                      ("dense_dir", str), ("k_dense", int), ("only_countries", str), ("max_df_name", float), ("max_df_addr", float), ("max_df_combo", float),
                      ("n_folds", int), ("n_predict_models", int), ("seed", int)):
        ap.add_argument("--" + name.replace("_", "-"), type=typ, default=None)
    a = ap.parse_args()
    cfg = Config(a.data_dir, a.work_dir, a.out_dir)
    for k in ("n_jobs", "train_frac", "k_name", "k_addr", "k_combo", "k_reverse", "exact_name_cap", "k_region", "cand_min_p",
              "dense_dir", "k_dense", "only_countries", "max_df_name", "max_df_addr",
              "max_df_combo", "n_folds", "n_predict_models", "seed"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    io_utils.ensure_dir(cfg.work_dir)
    io_utils.ensure_dir(cfg.out_dir)
    log(f"config: {asdict(cfg)}", cfg)
    if a.stage in ("prepare", "all"):
        stage_prepare(cfg)
    if a.stage == "regions" or (a.stage == "all" and cfg.k_region > 0):
        stage_regions(cfg)
    if a.stage in ("pairs", "all"):
        stage_pairs(cfg)
    if a.stage == "prune" or (a.stage == "all" and cfg.prune):
        stage_prune(cfg)
    if a.stage == "augment":
        stage_augment(cfg)
    if a.stage == "augment_ce":
        stage_augment_ce(cfg, a.ce_dir, a.ce_prefix)
        ce2 = os.path.join(cfg.work_dir, "ce2")  # optional 2nd cross-encoder (mDeBERTa), merged when present
        if a.ce_prefix == "ce" and all(os.path.exists(os.path.join(ce2, f"{sp}.parquet")) for sp in ("train", "test")):
            log(f"found {ce2}: merging second cross-encoder scores as ce2_*", cfg)
            stage_augment_ce(cfg, ce2, "ce2")
    if a.stage in ("train", "all"):
        stage_train(cfg)
    if a.stage in ("predict", "all"):
        stage_predict(cfg)
    log("finished", cfg)


if __name__ == "__main__":
    main()
