# Experiment 2: Widening the Cross-Encoder Band (`[0.001, 0.999]`)

**Author:** L2 (Inference & Retrieval Scaling)  
**Date:** 2026-09-27 ~14:45 IST  
**Baseline:** Run 6 (Validation F0.5 = 0.98769, Public LB = 0.982608, e5 Band = `[0.005, 0.995]`)  
**Target:** Eliminate `ce_p = NaN` blind spots on uncertain pairs; capture **+0.002 to +0.004** F0.5  
**Hardware Strategy:** L1 handles band extraction & retraining; L2 handles GPU scoring on Kaggle T4 x2 (~15–20 min)  

---

## 1. Executive Summary

In Run 6, the cross-encoder feature `ce_p` (`intfloat/multilingual-e5-small`) proved to be the **3rd most important feature in the entire pipeline** (1.4M gain in LightGBM, surpassed only by `pre_p` and `ce2_p`).

However, to control scoring time during earlier runs, `ce_band6.py` restricted cross-encoder scoring to pairs whose pre-ranker probability fell strictly inside:
$$\text{pre\_p} \in [0.005, 0.995]$$

### The Problem:
1. **The NaN Blind Spot:** For any candidate pair falling outside this band (especially marginal pairs with $\text{pre\_p} \in [0.001, 0.005]$), `ce_p` was assigned `NaN`.
2. **False Negatives in Borderline Region:** Run 4 and Run 6 error analyses revealed that many false negatives were genuinely uncertain pairs whose pre-ranker score was low, but whose true text similarity was high. Because `ce_p` was missing, LightGBM had to judge them without its strongest semantic signal.
3. **Manageable Load:** In Run 6, candidate pruning already trimmed pairs to **8.34 per S1 entity**. Widening the band to `[0.001, 0.999]` adds only a moderate number of pairs, which earlier runs never scored.

- **Expected Lift:** Validation $F_{0.5}$ **+0.002 to +0.004**, Public LB **+0.002 to +0.004**.
- **Kaggle Execution Time:** **~15 to 20 minutes** on dual Tesla T4 GPUs.

---

## 2. Division of Labor (L1 ↔ L2)

Because the 40 GB intermediate files reside on L1, the workload is partitioned for zero waste:

```
[L1: Adarsh] ce_band6.py band (0.001, 0.999) ---> todo_train.parquet (~15 MB) ---+
                                             ---> todo_test.parquet  (~30 MB) ---|
                                                                                  v
                                                        [L2: Akshaya on Kaggle T4x2]
                                                        Cross-Encoder Scoring (~15 min)
                                                                                  |
[L1: Adarsh] ce_band6.py merge <----------------- scored_train.parquet <----------+
             --stage augment_ce --stage train <-- scored_test.parquet
             OOF Gate Check & Submit!
```

---

## 3. Step-by-Step Instructions for L1 (Adarsh): Step 1

On Laptop L1, open PowerShell in the project directory and run:

```powershell
python dev_run6/scripts/ce_band6.py band C:/Users/ANEXUS/Downloads/PsychicLearn_work6 0.001 0.999
```

### What This Does:
- Reuses all cached scores from Run 4, Run 5, and Run 6.
- Identifies newly uncovered candidate pairs in the widened range `[0.001, 0.999]`.
- Exports:
  - `C:/Users/ANEXUS/Downloads/PsychicLearn_work6/ce/todo_train.parquet`
  - `C:/Users/ANEXUS/Downloads/PsychicLearn_work6/ce/todo_test.parquet`
- Adarsh sends these two files to L2 (or attaches them to GitHub release `from-L1`).

---

## 4. Step-by-Step Instructions for L2 (Akshaya): Kaggle Scoring

### Step 1: Notebook Setup on Kaggle
1. Go to [kaggle.com](https://www.kaggle.com) → **+ Create → New Notebook**.
2. Rename to: `psychiclearn-exp2-ce-scoring`.
3. Right sidebar settings:
   - **Accelerator:** `GPU T4 x2`
   - **Internet:** `On`
4. Attach Inputs via **`+ Add Input`**:
   - `psychiclearn-raw` (contains `dataset/test/` and `dataset/train/` TSVs)
   - Upload `todo_train.parquet` and `todo_test.parquet` as a private dataset (`psychiclearn-exp2-todo`) or attach from previous inputs.

---

### Step 2: Paste the Fast Scoring Script into Kaggle Cell 1

Paste and execute this complete script:

```python
import os, glob, time, sys
import numpy as np
import pandas as pd
import torch
import pyarrow.csv as pv, pyarrow as pa
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("="*60)
print("EXPERIMENT 2: HIGH-THROUGHPUT CROSS-ENCODER SCORING")
print("="*60)

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH = 512
MAX_LEN = 96
OUT_DIR = "/kaggle/working"

# 1. Locate Dataset & Todo files
def find_file(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    hits = [h for h in hits if "__MACOSX" not in h]
    assert hits, f"File not found for pattern: {pattern}"
    return hits[0]

DATA_DIR = os.path.dirname(find_file("/kaggle/input/**/train_source1.tsv"))
TODO_TRAIN = find_file("/kaggle/input/**/todo_train.parquet")
TODO_TEST = find_file("/kaggle/input/**/todo_test.parquet")

print(f"Data Dir:   {DATA_DIR}")
print(f"Todo Train: {TODO_TRAIN}")
print(f"Todo Test:  {TODO_TEST}")

# 2. Setup Dual GPUs
n_gpu = torch.cuda.device_count()
print(f"GPUs available: {n_gpu} {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).cuda()
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
model.eval()

# 3. Helper: Read Text Lookup
def read_raw_texts(data_dir, split, needed_ids):
    needed = set(needed_ids)
    texts = {}
    for src in (1, 2, 3):
        fn = os.path.join(data_dir, f"{split}_source{src}.tsv")
        tb = pv.read_csv(fn, parse_options=pv.ParseOptions(delimiter="\t"),
                         convert_options=pv.ConvertOptions(include_columns=["entity_id", "business_name", "business_address"]))
        df = tb.to_pandas()
        df = df[df["entity_id"].isin(needed)]
        t = (df["business_name"].fillna("") + " ; " + df["business_address"].fillna("")).str.slice(0, 300)
        texts.update(dict(zip(df["entity_id"], t)))
        del df, tb
    return texts

# 4. Fast Batched Inference
@torch.no_grad()
def score_pairs(df_pairs, split):
    print(f"\nScoring {split}: {len(df_pairs):,} pairs...")
    s1_ids = df_pairs["s1_id"].values
    cand_ids = df_pairs["cand_id"].values
    
    needed_ids = set(s1_ids) | set(cand_ids)
    print(f"Loading raw text for {len(needed_ids):,} entities...")
    text_map = read_raw_texts(DATA_DIR, split, needed_ids)
    
    a_txt = [text_map.get(s, "") for s in s1_ids]
    b_txt = [text_map.get(c, "") for c in cand_ids]
    del text_map
    
    # Sort by text length to minimize padding overhead
    order = np.argsort([len(x) + len(y) for x, y in zip(a_txt, b_txt)])
    out_probs = np.zeros(len(df_pairs), dtype=np.float32)
    
    t0 = time.time()
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        batch_a = [a_txt[i] for i in idx]
        batch_b = [b_txt[i] for i in idx]
        
        enc = tok(batch_a, batch_b, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.squeeze(-1)
            probs = torch.sigmoid(logits)
        out_probs[idx] = probs.cpu().numpy()
        
        if (s // BATCH) % 500 == 0 and s > 0:
            rate = s / max(time.time() - t0, 1e-6)
            eta = (len(order) - s) / rate / 60
            print(f"  [{s:,}/{len(order):,}] - {rate:.0f} pairs/sec - ETA: {eta:.1f} min")
            
    df_result = df_pairs.copy()
    df_result["ce_p"] = out_probs
    out_path = os.path.join(OUT_DIR, f"scored_{split}.parquet")
    df_result.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} ({len(df_result):,} rows, mean score: {out_probs.mean():.4f}) in {(time.time() - t0)/60:.1f} min")
    return out_path

# 5. Execute
df_todo_train = pd.read_parquet(TODO_TRAIN)
df_todo_test = pd.read_parquet(TODO_TEST)

score_pairs(df_todo_train, "train")
score_pairs(df_todo_test, "test")

print("\n" + "="*60)
print("EXPERIMENT 2 SCORING COMPLETE! Ready to download from Output.")
print("="*60)
```

### Step 3: Download & Handover
1. Go to the notebook's **Output** tab and download:
   - `scored_train.parquet`
   - `scored_test.parquet`
2. Upload both files to GitHub Release **`from-L2`** (or send directly to Adarsh).

---

## 5. Step-by-Step Instructions for L1 (Adarsh): Step 2

Once Adarsh receives `scored_train.parquet` and `scored_test.parquet`, he copies them into `PsychicLearn_work6/ce/` and executes on L1:

```powershell
# 1. Merge new scores with cached scores
python dev_run6/scripts/ce_band6.py merge C:/Users/ANEXUS/Downloads/PsychicLearn_work6

# 2. Re-run augmentation, training, and prediction
python -m ber.pipeline --data-dir C:/Users/ANEXUS/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset `
  --work-dir C:/Users/ANEXUS/Downloads/PsychicLearn_work6 `
  --out-dir C:/Users/ANEXUS/Downloads/PsychicLearn_run7_output `
  --stage augment_ce --ce-dir C:/Users/ANEXUS/Downloads/PsychicLearn_work6/ce `
  --stage train --stage predict
```

---

## 6. Evaluation Gate: Deciding Whether to Upload

In `PsychicLearn_work6/report_train.json`:
- **Run 6 Baseline:** OOF Macro $F_{0.5} = \mathbf{0.98769}$
- **Upload Gate:** If OOF Macro $F_{0.5} \ge \mathbf{0.98820}$ ($+0.0005$ lift), validate and upload `matching_results.tsv` as **Submission #7**.
- **Expected Score:** **LB 0.985 to 0.987** on India/US alone, and **~0.989 to 0.991** when combined with the France threshold variant (Exp 1).
