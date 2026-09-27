"""France-only threshold variant from saved test probabilities (no retraining).

  stats <work>                  : France/India/US predicted-match stats for a grid of France thresholds
  write <work> <out> <t1> <t2>  : matching_results.tsv with France thresholds (t1, t2), all other countries
                                  unchanged; candidate_pairs.tsv is copied from <work>'s run output

Only France predictions change, so the leaderboard difference to the standard file is purely the
France effect (France has no labels, so this is the only way to measure a France-only change).
"""
import json, os, shutil, sys
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_src = os.path.abspath(os.path.join(script_dir, "..", "src"))
sys.path.insert(0, repo_src)
sys.path.insert(0, r"C:\Users\ANEXUS\Downloads\PsychicLearn_dev\src")
from ber import decide, io_utils, writer

cmd, W = sys.argv[1], sys.argv[2]
rep = json.load(open(os.path.join(W, "report_train.json")))
T1, T2 = rep["t1"], rep["t2"]
DATA = rep.get("config", {}).get("data_dir", r"C:\Users\ANEXUS\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset")
df = pd.read_parquet(os.path.join(W, "test_pred.parquet"), columns=["s1_id", "cand_id", "p"])
s1 = pd.read_parquet(os.path.join(W, "norm", "test_s1.parquet"), columns=["entity_id", "country"])
country = s1.set_index("entity_id")["country"]
df["country"] = df["s1_id"].map(country).values
n_s1 = s1["country"].value_counts().to_dict()


def run(t1f, t2f):
    pred = {}
    for c, g in df.groupby("country"):
        a, b = (t1f, t2f) if c == "France" else (T1, T2)
        pred.update(decide.decide(g, a, b))
    return pred


def stats(pred):
    out = {}
    for c in ("France", "India", "US"):
        ids = s1.loc[s1["country"] == c, "entity_id"]
        n = ids.map(lambda s: len(pred.get(s, ()))).values
        out[c] = (round(float(n.mean()), 3), f"{(n == 0).mean():.1%}")
    return out


if cmd == "stats":
    print(f"standard t1={T1} t2={T2}: {stats(run(T1, T2))}", flush=True)
    for d in (0.05, 0.10, 0.15, 0.20):
        a, b = round(T1 - d, 3), round(T2 - d, 3)
        print(f"France t1={a} t2={b}: {stats(run(a, b))}", flush=True)
else:
    out, t1f, t2f = sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
    os.makedirs(out, exist_ok=True)
    pred = run(t1f, t2f)
    ids = io_utils.read_source(DATA, "test", 1)["entity_id"].tolist()
    writer.write_id_lists(os.path.join(out, "matching_results.tsv"), "matched_entity_ids", ids, pred)
    src_out = rep["config"]["out_dir"]
    cand_path = os.path.join(src_out, "candidate_pairs.tsv")
    if os.path.exists(cand_path):
        shutil.copy(cand_path, os.path.join(out, "candidate_pairs.tsv"))
    print(f"written {out} with France t1={t1f} t2={t2f}: {stats(pred)}", flush=True)
