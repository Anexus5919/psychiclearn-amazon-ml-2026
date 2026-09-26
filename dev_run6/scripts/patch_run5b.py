import io, os, py_compile
p = r"C:\Users\ANEXUS\Downloads\PsychicLearn_dev\src\ber\pipeline.py"
s = io.open(p, encoding="utf-8").read()
rep = [
('''CE_COLS = ["ce_p", "ce_rank", "ce_gap", "ce_n50"]
''',
'''CE_COLS = ["ce_p", "ce_rank", "ce_gap", "ce_n50"]
CE2_COLS = ["ce2_p", "ce2_rank", "ce2_gap", "ce2_n50"]  # second cross-encoder (mDeBERTa), optional
'''),
('''        feats += [c for c in fx.EXTRA + CE_COLS if c in names]''',
'''        feats += [c for c in fx.EXTRA + CE_COLS + CE2_COLS if c in names]'''),
('''def stage_augment_ce(cfg, ce_dir):
    """Merge cross-encoder scores (ce_dir/<split>.parquet: s1_id, cand_id, ce_p) into the pruned files;
    unscored pairs get NaN. Adds per-entity rank / gap-to-best / count>=0.5 of the CE score."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    for split in ("train", "test"):
        ce = pd.read_parquet(os.path.join(ce_dir, f"{split}.parquet"))
        for country, files in pair_files(cfg, split, "pairs_pruned").items():
            path = files[0]
            df = pd.read_parquet(path)
            df = df.drop(columns=[c for c in CE_COLS if c in df.columns])
            df = df.merge(ce, on=["s1_id", "cand_id"], how="left")
            g = df.groupby("s1_id")["ce_p"]
            df["ce_rank"] = g.rank(ascending=False, method="min").astype(np.float32)
            df["ce_gap"] = (g.transform("max") - df["ce_p"]).astype(np.float32)
            df["ce_n50"] = (df["ce_p"] >= 0.5).astype(np.float32).groupby(df["s1_id"]).transform("sum")''',
'''def stage_augment_ce(cfg, ce_dir, prefix="ce"):
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
            df[cols[3]] = (df[cols[0]] >= 0.5).astype(np.float32).groupby(df["s1_id"]).transform("sum")'''),
('''            log(f"  ce scores merged: {split} {country}", cfg)''',
'''            log(f"  {prefix} scores merged: {split} {country}", cfg)'''),
('''    df = pd.concat(scored, ignore_index=True)
    pred = decide.decide(df, rep["t1"], rep["t2"])''',
'''    df = pd.concat(scored, ignore_index=True)
    df.to_parquet(os.path.join(cfg.work_dir, "test_pred.parquet"), index=False)  # p >= 0.01 (pseudo-labels, analysis)
    pred = decide.decide(df, rep["t1"], rep["t2"])'''),
('''    ap.add_argument("--ce-dir", default=None, help="directory with cross-encoder scores (train.parquet, test.parquet)")''',
'''    ap.add_argument("--ce-dir", default=None, help="directory with cross-encoder scores (train.parquet, test.parquet)")
    ap.add_argument("--ce-prefix", default="ce", help="feature prefix for --stage augment_ce: ce (e5) or ce2 (mDeBERTa)")'''),
('''        stage_augment_ce(cfg, a.ce_dir)''',
'''        stage_augment_ce(cfg, a.ce_dir, a.ce_prefix)'''),
]
for a, b in rep:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
io.open(p + ".new", "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(p + ".new", doraise=True)
os.replace(p + ".new", p)
print("ok")
