# WS8: Dense Retrieval — Run Log

**Branch:** `feat/ws8-dense-retrieval`  
**Owner:** L2  
**Started:** 2026-09-26 ~18:38 IST  
**Guide:** `kaggle/DENSE_GUIDE_L2.md`  
**Script:** `kaggle/dense_retrieval.py` (unchanged except `SMOKE` flag during smoke test; reverted before real run)

---

## Script parameters (real run — SMOKE = 0)

| Parameter | Value |
|---|---|
| `MODEL` | `intfloat/multilingual-e5-small` (MIT, 118M params) |
| `MAXLEN` | 64 |
| `K` | 15 (top-15 nearest neighbours per S1 record) |
| `N_TRAIN_PAIRS` | 2,000,000 |
| `BATCH` | 256 |
| `LR` | 3e-5 |
| `TAU` | 0.05 |
| `SMOKE` | **0** (real run) |

## Kaggle notebook settings

| Setting | Value |
|---|---|
| Accelerator | GPU T4 x2 |
| Internet | On (model download) |
| Version name | `dense run5` |

## Inputs (from release `from-L1`)

| File | Source |
|---|---|
| `6ab10eb3b23ba_student_resource.zip` | challenge portal or release `from-L1` → Assets |
| `train_queries.parquet` | release `from-L1` → Assets |

Both dragged into private dataset `psychiclearn-raw` on Kaggle (Private, competition data).

## Fine-tuning exclusion

The 304,555 validation businesses in `train_queries.parquet` are excluded from fine-tuning
(`gt = gt[... & ~gt["source1_entity_id"].isin(queries)]`), so recall gain measured on them is honest.

## Expected outputs (/kaggle/working)

| File | Approx size |
|---|---|
| `dense_train.parquet` | 60-100 MB |
| `dense_test.parquet` | 300-500 MB |

Columns: `s1_id, cand_id, dense_cos, dense_rank` (top-15 per S1 record, per country).

## Smoke test results

*(To be filled in by L2 after running SMOKE=1 on Kaggle)*

- [ ] `GPUs: 2` confirmed
- [ ] `fine-tuning on 20,000 pairs` shown
- [ ] `fine-tuning done` shown
- [ ] Country lines for train India/US and test France/India/US shown
- [ ] `ALL DONE (SMOKE TEST - outputs are NOT usable; set SMOKE = 0)` shown

## Real run results

*(To be filled in by L2 after the run completes ~2-2.5 h)*

| Metric | Value |
|---|---|
| Fine-tuning loss (start to end) | |
| Time to fine-tune | |
| Time total | |
| `dense_train.parquet` size | |
| `dense_test.parquet` size | |

## Handover

- Upload `dense_train.parquet` and `dense_test.parquet` to release `from-L2`.
- Tell Adarsh (L1).
- L1 gate: share of run-5 missed true pairs recovered by dense top-15 on validation businesses.
  Run 6 = run 5 + dense pass, uploaded only if OOF beats run 5.

## Troubleshooting (from DENSE_GUIDE_L2.md)

| Problem | Fix |
|---|---|
| `input file not found: train_source1.tsv` | Dataset not attached or zip not unpacked. Check Input panel. |
| `input file not found: train_queries.parquet` | Add file to dataset (New Version). |
| `GPUs: 1` or `0` | Accelerator isn't GPU T4 x2. |
| `CUDA out of memory` during fine-tuning | Change `BATCH = 256` to `128`. |
| `CUDA out of memory` in neighbour search | Change `def knn(q, p, k, chunk=512)` to `chunk=256`. |
| Model download error | Internet is off. |
