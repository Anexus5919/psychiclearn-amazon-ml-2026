# L4 Guide: Moderated Data Scaling (`train_frac = 0.15`) on Kaggle

**Who this is for:** L4 (RTX 4060 laptop / Kaggle expert).  
**Objective:** Scale the LightGBM training pool from 8% to **15% of businesses** (~570,000 entities, ~13M candidate pairs) to expose the ranker to more hard distractors, edge-case names, and regional address variants.  
**Why Option A:** Keeps RAM under 16 GB (well within Kaggle's 30 GB limit) and completes in **~2.5–3 hours** (finishing by ~17:00–17:30 IST, safely ahead of the 20:00 IST freeze deadline).  
**Expected Gain:** Validation $F_{0.5}$ **+0.001 to +0.002**, Public LB **+0.001 to +0.003**.  

---

## 1. Prerequisites & Resource Checklist

- **Environment:** Kaggle Notebook (free tier).
- **Accelerator:** **GPU T4 x2** (used for cross-encoder scoring and rapid batch operations).
- **RAM:** Standard GPU instance gives **30.0 GB RAM** (required for ~13M LightGBM dataset).
- **Disk:** 73 GB scratch space in `/kaggle/working`.
- **Attached Inputs:**
  1. `psychiclearn-raw` (Private dataset containing `dataset/` and competition TSVs).
  2. Your previous notebook output (`psychiclearn-mdeberta` or `dense run5` for cached cross-encoder/dense weights).

---

## 2. Step-by-Step Kaggle Setup

### Step 1: Create the Notebook
1. Go to [kaggle.com](https://www.kaggle.com) → **+ Create → New Notebook**.
2. Rename notebook to: `psychiclearn-exp6-train15`.
3. In the right sidebar:
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** `On` (needed for base HuggingFace tokenizer downloads if not cached)
4. Under **Input → Add Input**:
   - Add your competition dataset: `psychiclearn-raw`
   - Add your previous run: `psychiclearn-mdeberta` (for `mdeberta_ce/` weights)
   - Add L2's dense output: `dense run5` (for `dense_test.parquet` and `dense_train.parquet`)

---

### Step 2: Paste the Pipeline Runner into Cell 1

Paste the following automated execution script into the notebook cell:

```python
import os, sys, glob, shutil, subprocess, time

print("="*60)
print("EXPERIMENT 6: MODERATED DATA SCALING (train_frac = 0.15)")
print("="*60)

# 1. Environment and Paths
ROOT = "/kaggle/working"
SRC = os.path.join(ROOT, "src")
DATA = None

# Locate dataset root
for p in ["/kaggle/input/**/dataset", "/kaggle/input/**/train_source1.tsv"]:
    hits = glob.glob(p, recursive=True)
    if hits:
        DATA = os.path.dirname(hits[0]) if "train_source1.tsv" in hits[0] else hits[0]
        break
assert DATA, "Dataset not found! Ensure psychiclearn-raw is attached."
print(f"Dataset located at: {DATA}")

# Locate Dense files
DENSE_DIR = None
for p in ["/kaggle/input/**/dense_train.parquet"]:
    hits = glob.glob(p, recursive=True)
    if hits:
        DENSE_DIR = os.path.dirname(hits[0])
        break
print(f"Dense directory: {DENSE_DIR}")

# Locate mDeBERTa cross-encoder model
CE2_MODEL = None
for p in ["/kaggle/input/**/mdeberta_ce"]:
    hits = glob.glob(p, recursive=True)
    if hits:
        CE2_MODEL = hits[0]
        break
print(f"mDeBERTa model: {CE2_MODEL}")

# 2. Extract repository source code
os.makedirs(SRC, exist_ok=True)
# Clone or copy src if in input
repo_hits = glob.glob("/kaggle/input/**/dev_run6/src", recursive=True)
if repo_hits:
    shutil.copytree(repo_hits[0], os.path.join(SRC, "ber"), dirs_exist_ok=True)
else:
    # Clone private repo via shallow git if token provided, or pull dev_run6
    !git clone --depth 1 https://github.com/Anexus5919/psychiclearn-amazon-ml-2026.git repo
    shutil.copytree("repo/dev_run6/src/ber", os.path.join(SRC, "ber"), dirs_exist_ok=True)

sys.path.insert(0, SRC)
from ber import pipeline

# 3. Pipeline Configuration for Option A
WORK = os.path.join(ROOT, "work_exp6")
OUT = os.path.join(ROOT, "output_exp6")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

cfg_args = [
    sys.executable, "-m", "ber.pipeline",
    "--data-dir", DATA,
    "--work-dir", WORK,
    "--out-dir", OUT,
    "--train-frac", "0.15",          # <--- OPTION A: 15% OF BUSINESSES (~570k S1 entities)
    "--exact-name-cap", "50",
    "--k-region", "10",
    "--k-name", "10",
    "--k-addr", "10",
    "--k-combo", "15",
    "--cand-min-p", "0.0001",
    "--n-jobs", "4"
]

if DENSE_DIR:
    cfg_args.extend(["--dense-dir", DENSE_DIR, "--k-dense", "15"])

# 4. Execute Stages sequentially
stages = ["prepare", "regions", "pairs", "prune", "augment", "train", "predict"]

for stg in stages:
    t0 = time.time()
    print(f"\n>>> STARTING STAGE: {stg.upper()} at {time.strftime('%H:%M:%S')} <<<")
    cmd = cfg_args + ["--stage", stg]
    ret = subprocess.run(cmd, cwd=SRC)
    if ret.returncode != 0:
        print(f"FAILED stage {stg}! Aborting.")
        sys.exit(1)
    print(f">>> FINISHED STAGE: {stg.upper()} in {(time.time() - t0)/60:.1f} min <<<")

print("\n" + "="*60)
print("EXPERIMENT 6 OPTION A COMPLETE!")
print("="*60)
```

---

## 3. Runtime & Progress Expectations

| Stage | Estimated Time | What is Happening |
|---|---|---|
| **`prepare` & `regions`** | ~5 min | Normalizes text and extracts coarse location keys. |
| **`pairs`** | ~40–50 min | Generates candidate pairs for 15% of training businesses (~570k entities). |
| **`prune`** | ~15–20 min | Applies pre-ranker to reduce candidate count to ~8.3 pairs per entity (~13M total). |
| **`augment`** | ~30 min | Computes base lexical, token overlap, and cross-features. |
| **`train`** | ~45–60 min | LightGBM 4-fold grouped cross-validation on 13M rows (RAM usage ~14–16 GB). |
| **`predict`** | ~15 min | Scores test set and outputs `matching_results.tsv` and `report_train.json`. |
| **Total Runtime** | **~2.5 to 3.0 h** | **Finishes by ~17:00–17:30 IST.** |

---

## 4. Evaluation Gate: Deciding Whether to Upload

Once the run completes, inspect `report_train.json` in `/kaggle/working/work_exp6/`:

1. **Check OOF Macro $F_{0.5}$:**
   - Run 6 Baseline: **`0.98769`**
   - **Upload Gate:** OOF $F_{0.5} \ge \mathbf{0.98820}$ ($+0.0005$ lift).
2. **Check per-country validation:**
   - India: should be $\ge 0.9882$
   - US: should be $\ge 0.9882$

---

## 5. Handover to Adarsh (L1)

If the gate is passed ($\ge 0.9882$):
1. Download from the Kaggle Output tab:
   - `matching_results.tsv`
   - `report_train.json`
   - `models/fold*.txt` (the trained 15% LightGBM boosters)
2. Upload them to GitHub Releases under **`from-L4`**.
3. Notify Adarsh (L1) to run the submission validator and submit as **Submission #7**.
