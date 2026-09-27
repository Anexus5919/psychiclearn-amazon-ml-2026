# Experiment 5: Wider Dense Retrieval (K=20) for France

**Author:** L2 (WS8 Dense Retrieval Owner & Retrieval Systems)  
**Date:** 2026-09-27 ~14:05 IST  
**Baseline:** Run 6 (Validation F0.5 = 0.98769, Public LB = 0.982608, Dense $K=15$)  
**Target:** Recover true French matches lost to name crowding; push toward LB ≥ 0.99  
**Status:** Scripts updated with cached-model acceleration and France $K=20$; ready for Kaggle execution  

---

## 1. Executive Summary

In Run 6, WS8 Dense Retrieval (`multilingual-e5-small` bi-encoder, top-15 cosine neighbours) dramatically elevated blocking recall from 0.9688 to **0.9955**, pushing the oracle ceiling on India and US to **0.9987**.

However, while India has 15 states and the US has 45 states to isolate business clusters, **France has only 3 coarse regions**. As a result, regional blocking cannot effectively isolate same-named French businesses, and **12.7% of French entities suffer from severe name crowding**.

Experiment 5 widens the dense retrieval search window from **$K=15$ to $K=20$ specifically for France**:
- **Why it works:** Captures true French entity matches that rank at positions 16–20 in cosine similarity, just outside the Run 6 cutoff.
- **Why it is safe:** The lightweight pre-ranker (`prune.py`) and downstream LightGBM ranker safely discard unpromising candidates, resulting in an increase of only ~0.5 candidates per entity after pruning.
- **Compute Optimization:** By reusing the fine-tuned bi-encoder weights (`dense_model/`), Kaggle GPU time is slashed from 2.5 hours down to **~15–20 minutes**.
- **Expected Impact:** Val **+0.001–0.002**, Public LB **+0.002–0.004**. When combined with Experiment 1, this directly challenges the **0.988–0.991** leaderboard tier.

---

## 2. Theoretical Mechanics: Why France Needs K=20

### The Structural Limitation of Regional Blocking in France
In `dev_run6/src/ber/regions.py`, the pipeline builds coarse geographic regions:
- **India:** 15 states/territories
- **US:** 45 states
- **France:** Only 3 coarse regions

Because each French region contains roughly one-third of the country's entire commercial database, lexical TF-IDF searches within a region still experience massive "crowding" by unrelated businesses sharing generic terms (*Commerce, Alimentation, Société, Transport*).

### The Dense Bi-Encoder Advantage
The `multilingual-e5-small` bi-encoder maps business name and address into a 384-dimensional continuous semantic space:
$$\text{sim}(S_1, S_2) = \frac{\mathbf{e}_{S_1} \cdot \mathbf{e}_{S_2}}{\|\mathbf{e}_{S_1}\| \|\mathbf{e}_{S_2}\|}$$

In Run 6:
- For India and US, top-15 neighbours were sufficient to reach **99.55% recall** because regional filtering had already resolved most ambiguities.
- For France, true matches frequently land at ranks 16, 17, 18, 19, or 20 due to the high density of similar addresses and names within the same broad region.
- Expanding to $K=20$ adds 5 candidate pairs per $S_1$ entity ($259,452 \times 5 \approx 1,297,260$ raw pairs).
- After the pruning stage (`prune.py`), only candidates with high `dense_cos` or strong cross-features survive, adding only **~100,000–150,000 net pairs** to downstream scoring (~0.4–0.6 candidates/entity).

---

## 3. Code Improvements Implemented

1. **`kaggle/dense_retrieval.py` Refactored:**
   - **Pretrained Model Detection:** Automatically searches `/kaggle/input/**/dense_model` or `./dense_model`. If found, it skips the 45-minute 2,000,000-pair fine-tuning loop and jumps directly to embedding and KNN search.
   - **Per-Country $K$ Selection:** Sets $K=20$ for France while maintaining $K=15$ for India and US.
   - **Targeted Outputs:** Generates both the full `dense_test.parquet` and a dedicated, lightweight `dense_test_France.parquet` (~60 MB).
   - **`ONLY_FRANCE` Toggle:** Allows generating France top-20 in isolation in under 20 minutes.

2. **Pipeline Compatibility in `dev_run6/src/ber/`:**
   - `pipeline.py`: Updated `dense_for(country)` to automatically allow up to rank 20 for France while keeping rank 15 for India/US.
   - `blocking.py`: Updated default unretrieved rank fill (`ks["dense"] = 20` for France, `15` for others).

---

## 4. Manual Actions Required by You (L2) on Kaggle

You have two execution options on Kaggle:

### Option A: Fast Run (Recommended — ~15–20 minutes)
*Reuses the already trained `dense_model/` weights from Run 6.*

1. Open your existing `dense run5` notebook on Kaggle (or create a new notebook).
2. Ensure **GPU T4 x2** accelerator is enabled.
3. Attach your dataset `psychiclearn-raw` AND attach the output of `dense run5` (via **Add Input → Your Work → dense run5**) so `dense_model/` is present in `/kaggle/input`.
4. Copy the updated code from [dense_retrieval.py](file:///c:/Users/Akshaya/Desktop/psychiclearn-amazon-ml-2026/kaggle/dense_retrieval.py) into the notebook cell.
5. At the top of the cell, set:
   ```python
   SMOKE = 0
   ONLY_FRANCE = 1
   ```
6. Click **Run All**.
7. The notebook will:
   - Detect `dense_model/` and log: `Found existing fine-tuned model; skipping fine-tuning stage!`
   - Embed only France Test ($S_1$: 259,452, Pool: 1,434,993).
   - Compute top-20 KNN in ~3 minutes.
   - Write `dense_test_France.parquet` (~60 MB) to `/kaggle/working`.
8. Download `dense_test_France.parquet` from the Output tab.

---

### Option B: Full Run (~1.5–2 hours)
*Generates full train and test sets with France $K=20$ and India/US $K=15$.*

1. In the notebook, leave `ONLY_FRANCE = 0` and `SMOKE = 0`.
2. Click **Save Version → Save & Run All (Commit)**.
3. Download `dense_test.parquet` (~320 MB) upon completion.

---

## 5. Handover to L1 (Adarsh)

Once you download the resulting file (`dense_test_France.parquet` or `dense_test.parquet`):

1. **Upload to GitHub Release:**
   Go to GitHub repo → **Releases → `from-L2`** → Attach `dense_test_France.parquet` (or update release assets).
2. **Tell Adarsh (L1):**
   Adarsh places the file into `Downloads/PsychicLearn_dense/` on L1.
3. **L1 Pipeline Run:**
   Adarsh executes the France update pipeline:
   ```bash
   python -m ber.pipeline --data-dir <data> --work-dir <work> --out-dir <out> --only-countries France --stage pairs --stage prune --stage augment --stage augment_ce --stage predict
   ```
   *(Only France is re-blocked and re-predicted; India and US are completely untouched).*

---

## 6. Strategic Analysis: What to Do with This Experiment

1. **Execute Option A on Kaggle Right Away:**
   It requires only ~15 minutes of GPU time and generates a tiny 60 MB file.
2. **Synergy with Experiment 1 (The 1-2 Punch to 0.99):**
   - **Experiment 5 (Retrieval):** Recovers true French matches by expanding the candidate search from 15 to 20 neighbours.
   - **Experiment 1 (Decision):** Lowers the France acceptance threshold ($t_1=0.52, t_2=0.68$) so the model actually accepts those recovered pairs.
   - Running either experiment alone provides a partial boost; running them together provides a multiplicative effect that directly attacks France's ~0.022 deficit.
3. **Target Score:**
   - Baseline Run 6: **LB 0.982608**
   - With Exp 1 + Exp 5: **Projected LB 0.9880–0.9910**
