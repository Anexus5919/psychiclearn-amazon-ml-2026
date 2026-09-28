"""One-command reproduction of our best submission (run 6, public leaderboard 0.982608).

Usage (from code/business_entity_resolution/src):
    python run_all.py --data-dir <dataset dir with train/ and test/> --work-root <scratch dir> --out-dir <output dir>
    python run_all.py ... --steps ce_score,mdeberta,augment,train,predict      # run / resume selected steps
    python run_all.py ... --smoke            # quick end-to-end check on a small dataset (see README)

Every step writes to --work-root and skips work that already exists, so an interrupted run resumes.
Steps marked [GPU] need a CUDA GPU (we used an RTX 3050 6 GB laptop and Kaggle T4 x2); the rest run on CPU.

  base      normalise every file (learns the Indic->Latin dictionary), base retrieval + learned pruning
            on the base training businesses (train split only; defines the pruning model used below)
  ce_data   candidate pairs for a separate 3% of businesses = cross-encoder training data
  ce_train  fine-tune multilingual-e5-small as a cross-encoder on those pairs                    [GPU]
  queries   final training/validation businesses (base set + 6.5%, disjoint from ce_data)
  dense     fine-tune multilingual-e5-small as a bi-encoder, top-15 neighbours for every S1     [GPU]
  regions   state / region key of every record
  pairs     retrieval: TF-IDF name / address / combined, exact name, region-restricted name, dense
  prune     learned pruning (keeps the fewest candidates that lose <= 0.2% of true matches)
  ce_score  e5 cross-encoder probability for candidates with pre-ranker p in [0.005, 0.995]      [GPU]
  mdeberta  fine-tune mDeBERTa-v3-base cross-encoder, score candidates with p in [0.02, 0.995]    [GPU]
  augment   extra + group features, then merge both cross-encoders' scores
  train     LightGBM, 4-fold CV grouped by business, thresholds tuned for macro F0.5
  predict   write matching_results.tsv + candidate_pairs.tsv (+ official validator if --validator)
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["base", "ce_data", "ce_train", "queries", "dense", "regions", "pairs", "prune", "ce_score",
         "mdeberta", "augment", "train", "predict"]
FINAL = ["--exact-name-cap", "50", "--k-region", "10"]          # final retrieval settings (all countries)
INDIA = ["--k-name", "15", "--k-addr", "15", "--k-combo", "20"]  # bigger search budget for India


def sh(args, env=None):
    print(">>", " ".join(str(a) for a in args), flush=True)
    e = dict(os.environ, PYTHONIOENCODING="utf-8", **(env or {}))
    subprocess.run([str(a) for a in args], cwd=HERE, env=e, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--validator", default=None, help="path to utils/validate_submission.py (optional)")
    ap.add_argument("--smoke", action="store_true", help="tiny settings for a quick end-to-end check")
    ap.add_argument("--mdeberta-model", default="microsoft/mdeberta-v3-base")
    a = ap.parse_args()
    py = sys.executable
    base, ce, fin = (os.path.join(a.work_root, d) for d in ("base", "ce", "final"))
    pipe = [py, "-m", "ber.pipeline", "--data-dir", a.data_dir]
    smoke = {"BER_SMOKE": "1"} if a.smoke else {}
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    for s in steps:
        assert s in STEPS, f"unknown step {s}"
    e5 = os.path.join(ce, "ce_e5small")

    if "base" in steps:
        sh(pipe + ["--work-dir", base, "--out-dir", os.path.join(base, "out"), "--stage", "prepare"])
        sh([py, "-c", "from ber import pipeline as p; c = p.Config(r'%s', r'%s', r'%s'); "
                      "p.stage_pairs(c, splits=('train',))" % (a.data_dir, base, os.path.join(base, "out"))])
        sh(pipe + ["--work-dir", base, "--out-dir", os.path.join(base, "out"), "--stage", "prune"])
    if "ce_data" in steps:
        sh([py, "-m", "ber.ce_data", "--data-dir", a.data_dir, "--src-work", base, "--ce-work", ce])
    if "ce_train" in steps and not os.path.exists(os.path.join(e5, "config.json")):
        extra = ["--max-pairs", "20000"] if a.smoke else []
        sh([py, "-m", "ber.cross_encoder", "train", "--data-dir", a.data_dir, "--work", ce, "--model-dir", e5,
            "--model", "intfloat/multilingual-e5-small", "--batch", "128", "--lr", "6e-5"] + extra)
    if "queries" in steps:
        sh([py, "-m", "ber.queries", "--base", base, "--ce", ce, "--final", fin])
    dense_dir = os.path.join(fin, "dense")
    if "dense" in steps and not os.path.exists(os.path.join(dense_dir, "dense_test.parquet")):
        din = os.path.join(a.work_root, "dense_in")
        os.makedirs(din, exist_ok=True)
        shutil.copy2(os.path.join(fin, "pairs", "train", "_queries.parquet"), os.path.join(din, "train_queries.parquet"))
        sh([py, os.path.join("gpu", "dense_retrieval.py")],
           env=dict(smoke, BER_IN=a.data_dir + os.pathsep + din, BER_OUT=dense_dir))
    run = pipe + ["--work-dir", fin, "--out-dir", a.out_dir] + FINAL + ["--dense-dir", dense_dir]
    if "regions" in steps:
        sh(run + ["--stage", "regions"])
    if "pairs" in steps:
        sh(run + ["--stage", "pairs", "--only-countries", "India"] + INDIA)
        sh(run + ["--stage", "pairs", "--only-countries", "US,France"])
    if "prune" in steps:
        sh(run + ["--stage", "prune"])
    if "ce_score" in steps:
        cdir = os.path.join(fin, "ce")
        os.makedirs(cdir, exist_ok=True)
        for sp in ("train", "test"):
            if os.path.exists(os.path.join(cdir, f"{sp}.parquet")):
                continue
            band = os.path.join(cdir, f"band_{sp}.parquet")
            sh([py, "-m", "ber.ce_export", "score", "--split", sp, "--lo", "0.005", "--hi", "0.995",
                "--data-dir", a.data_dir, "--work", fin, "--out", band])
            sh([py, "-m", "ber.cross_encoder", "score", "--data-dir", a.data_dir, "--split", sp, "--model-dir", e5,
                "--pairs", band, "--out", os.path.join(cdir, f"{sp}.parquet")])
    if "mdeberta" in steps and not os.path.exists(os.path.join(fin, "ce2", "test.parquet")):
        mi, mo, c2 = (os.path.join(fin, d) for d in ("mdeberta_in", "mdeberta_out", "ce2"))
        for d in (mi, mo, c2):
            os.makedirs(d, exist_ok=True)
        sh([py, "-m", "ber.ce_export", "train", "--data-dir", a.data_dir, "--work", ce,
            "--out", os.path.join(mi, "ce_train.parquet")])
        for sp in ("train", "test"):
            sh([py, "-m", "ber.ce_export", "score", "--split", sp, "--lo", "0.02", "--hi", "0.995",
                "--data-dir", a.data_dir, "--work", fin, "--out", os.path.join(mi, f"score_{sp}.parquet")])
        env = dict(BER_IN=mi, BER_OUT=mo, BER_MODEL=a.mdeberta_model)
        if a.smoke:
            env["BER_SMOKE"] = "3000"
        sh([py, os.path.join("gpu", "mdeberta_cross_encoder.py")], env=env)
        for sp in ("train", "test"):  # ce2/<split>.parquet is merged as ce2_* by the augment_ce stage
            shutil.copy2(os.path.join(mo, f"ce2_{sp}.parquet"), os.path.join(c2, f"{sp}.parquet"))
    if "augment" in steps:
        sh(run + ["--stage", "augment"])
        sh(run + ["--stage", "augment_ce", "--ce-dir", os.path.join(fin, "ce")])
    if "train" in steps:
        sh(run + ["--stage", "train"])
    if "predict" in steps:
        sh(run + ["--stage", "predict"])
        if a.validator:
            sh([py, a.validator, "--matching", os.path.join(a.out_dir, "matching_results.tsv"),
                "--candidate", os.path.join(a.out_dir, "candidate_pairs.tsv"),
                "--test-dir", os.path.join(a.data_dir, "test"), "--check-ids"])
    print("done:", ", ".join(steps), flush=True)


if __name__ == "__main__":
    main()
