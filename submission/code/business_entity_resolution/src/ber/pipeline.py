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
    seed: int = 42


T0 = time.time()


def log(msg, cfg=None):
    line = f"[{time.time() - T0:8.1f}s] {msg}"
    print(line, flush=True)
    if cfg is not None:
        with open(os.path.join(cfg.work_dir, "log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ------------------------------------------------------------------ stage 1: normalise
def stage_prepare(cfg):
    """Normalise every source file once and store the views as Parquet."""
    out = io_utils.ensure_dir(os.path.join(cfg.work_dir, "norm"))
    for split in ("train", "test"):
        for src in (1, 2, 3):
            path = os.path.join(out, f"{split}_s{src}.parquet")
            if os.path.exists(path):
                continue
            df = io_utils.read_source(cfg.data_dir, split, src)
            log(f"prepare {split} s{src}: {len(df):,} rows, countries={sorted(df['country'].unique())}", cfg)
            import pyarrow as pa
            import pyarrow.parquet as pq
            writer_ = None
            for chunk in normalize.iter_normalised(df, cfg.n_jobs):  # streamed to disk chunk by chunk
                tb = pa.Table.from_pandas(chunk[NORM_COLS], preserve_index=False)
                writer_ = writer_ or pq.ParquetWriter(path + ".tmp", tb.schema)
                writer_.write_table(tb)
            writer_.close()
            os.replace(path + ".tmp", path)
            del df
    gt_path = os.path.join(out, "train_gt.parquet")
    if not os.path.exists(gt_path):
        gt = io_utils.read_ground_truth(cfg.data_dir)
        pd.DataFrame({"s1_id": list(gt.keys()), "matched": [",".join(sorted(v)) for v in gt.values()]}).to_parquet(gt_path, index=False)
    log("prepare done", cfg)


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
        for country in countries:
            for src in (2, 3):
                path = os.path.join(out, f"{country}_s{src}.parquet")
                if os.path.exists(path):
                    continue
                s1 = load_norm(cfg, split, 1, country=country)
                pool = load_norm(cfg, split, src, country=country)
                qmask = s1["entity_id"].isin(queries).values
                log(f"pairs {split} {country} s{src}: S1={len(s1):,} (queries {qmask.sum():,}) pool={len(pool):,}", cfg)
                cand = blocking.block_partition(s1, pool, qmask, cfg, log=lambda m: log(m, cfg), tmp_dir=tmp_dir)
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


def pair_files(cfg, split):
    """{country: [pair files]} for a split (one file per country and target source)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(cfg.work_dir, "pairs", split, "*_s[23].parquet"))):
        out.setdefault(os.path.basename(f).rsplit("_s", 1)[0], []).append(f)
    return out


def iter_country_pairs(cfg, split, batch_rows=2_000_000):
    """Yield (keys_df, X) chunks per country with cross-source context filled in.

    ctx_rank_combo_all ranks each candidate by combo cosine across S2 and S3 together, so it is
    computed from the (small) key columns of both source files of a country before streaming the
    feature rows. Keys use Arrow-backed strings to keep memory low.
    """
    import pyarrow.parquet as pq
    fidx = features.FEATURES.index("ctx_rank_combo_all")
    for country, files in pair_files(cfg, split).items():
        keys = pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id", "cos_combo"], dtype_backend="pyarrow")
                          for f in files], ignore_index=True)
        ranks = keys.groupby("s1_id")["cos_combo"].rank(ascending=False, method="min").to_numpy(np.float32)
        off = 0
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=batch_rows, columns=features.FEATURES):
                X = np.column_stack([batch.column(c).to_numpy(zero_copy_only=False) for c in features.FEATURES]).astype(np.float32)
                n = X.shape[0]
                X[:, fidx] = ranks[off:off + n]
                yield country, keys.iloc[off:off + n][["s1_id", "cand_id"]].reset_index(drop=True), X
                off += n


# ------------------------------------------------------------------ stage 3: train + validate
def stage_train(cfg):
    """Out-of-fold LightGBM, threshold tuning and an end-to-end validation report."""
    gt = load_gt(cfg)
    import pyarrow.parquet as pq
    n_rows = sum(pq.ParquetFile(f).metadata.num_rows for fs in pair_files(cfg, "train").values() for f in fs)
    # disk-backed feature matrix: avoids holding (and briefly doubling) several GB in RAM
    X = np.lib.format.open_memmap(os.path.join(cfg.work_dir, "train_X.npy"), mode="w+", dtype=np.float32,
                                  shape=(n_rows, len(features.FEATURES)))
    parts, off = [], 0
    for _, k, x in iter_country_pairs(cfg, "train"):
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
    imp = pd.Series(models[0].feature_importance("gain"), index=features.FEATURES).sort_values(ascending=False)
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
    cands, scored, n_pairs = {}, [], 0
    for country, keys, X in iter_country_pairs(cfg, "test"):
        keys["p"] = model.predict(models, X, n_jobs=cfg.n_jobs)
        n_pairs += len(keys)
        scored.append(keys[keys["p"] >= 0.01])  # lower p can never be accepted (min threshold 0.02)
        # candidates stored as comma-joined chunks per S1 (compact); joined again when written
        for s1, joined in keys.groupby("s1_id", sort=False)["cand_id"].agg(",".join).items():
            cands.setdefault(s1, []).append(joined)
    log(f"test pairs scored: {n_pairs:,}", cfg)
    df = pd.concat(scored, ignore_index=True)
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
    ap.add_argument("--stage", default="all", choices=["prepare", "pairs", "train", "predict", "all"])
    for name, typ in (("n_jobs", int), ("train_frac", float), ("k_name", int), ("k_addr", int), ("k_combo", int),
                      ("k_reverse", int), ("max_df_name", float), ("max_df_addr", float), ("max_df_combo", float),
                      ("n_folds", int), ("n_predict_models", int), ("seed", int)):
        ap.add_argument("--" + name.replace("_", "-"), type=typ, default=None)
    a = ap.parse_args()
    cfg = Config(a.data_dir, a.work_dir, a.out_dir)
    for k in ("n_jobs", "train_frac", "k_name", "k_addr", "k_combo", "k_reverse", "max_df_name", "max_df_addr",
              "max_df_combo", "n_folds", "n_predict_models", "seed"):
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))
    io_utils.ensure_dir(cfg.work_dir)
    io_utils.ensure_dir(cfg.out_dir)
    log(f"config: {asdict(cfg)}", cfg)
    if a.stage in ("prepare", "all"):
        stage_prepare(cfg)
    if a.stage in ("pairs", "all"):
        stage_pairs(cfg)
    if a.stage in ("train", "all"):
        stage_train(cfg)
    if a.stage in ("predict", "all"):
        stage_predict(cfg)
    log("finished", cfg)


if __name__ == "__main__":
    main()
