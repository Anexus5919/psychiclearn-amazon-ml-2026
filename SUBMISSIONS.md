# Submission version history

This file follows the guideline: *"Maintain the version history of all your submissions, as
shortlisting will be based on the submitted solutions."*

Every leaderboard upload is recorded with:
- the **exact file** uploaded (SHA-256 of `matching_results.tsv`);
- the **archived copy** in `results/`, which decompresses to the same SHA-256 (verified);
- the **exact code** that produced it (folder + git tag);
- the **validation score** (out-of-fold macro F0.5 on held-out training businesses);
- the **public leaderboard** score.

Only the team leader uploads, with the files sent as-is. Every file passed the official
`utils/validate_submission.py --check-ids` before upload.

| # | Uploaded (IST) | Run | Validation F0.5 (India / US) | Public LB | SHA-256 of uploaded `matching_results.tsv` | Archived copy | Code | Git tag |
|---|---|---|---|---|---|---|---|---|
| 1 | 25 Sep 23:59 | Run 2 | 0.9515 (0.9196 / 0.9727) | **0.931** | `5c571c9177c45793e09d110a16481273c6c440ba67182403fc81cfe9ac018dc3` | `results/run2_matching_results_LB0.931.tsv.gz` | `submission/code/business_entity_resolution/` | `sub1-run2-LB0.931` |
| 2 | 26 Sep 02:50 | Run 3 | 0.9613 (0.9443 / 0.9726) | **0.941** | `3c1bc8120565cf13ac777c9a1c294739a9b21edbf6661b2eb09e8207ebae127f` | `results/run3_matching_results_LB0.941.tsv.gz` | `dev_run3/` | `sub2-run3-LB0.941` |
| 3 | 26 Sep 15:44 | Run 4 | 0.97628 (0.96710 / 0.98240) | **0.963** | `fa6be7823bbcdacd2c3aa01552cfeedb4897fe90df3ce8d9bae861d9ce790034` | `results/run4_matching_results_LB0.963.tsv.gz` | `dev_run4/` | `sub3-run4-LB0.963` |
| 4 | 26 Sep 23:45 | Run 5 | 0.97789 (0.96906 / 0.98378) | **0.967** | `e901975dd43178d1b5ae836c0d44bd2d64add672dd798fd2eb162fe27aea9a5e` | `results/run5_matching_results_LB0.967.tsv.gz` | `dev_run5/` | `sub4-run5-LB0.967` |
| 5 | 27 Sep 03:18 | Run 6 | 0.98769 (0.98772 / 0.98767) | **0.983** | `3f1e43a2b5cc52f7354efbd83efd47ed9b4049b82600a5465d8586cf85512e0c` | `results/run6_matching_results_LB0.983.tsv.gz` | `dev_run6/` | `sub5-run6-LB0.983` |

## What changed in each submission

| # | Change vs. the previous submission | Details |
|---|---|---|
| 1 | Baseline pipeline: normalisation → TF-IDF retrieval (name / address / combined) → ~50 features → LightGBM (4-fold grouped CV) → exclusive assignment + rank thresholds | `JOURNEY.md` §4–5 |
| 2 | + learned Indic→Latin transliteration (India); + learned candidate pruning (54 → 18 candidates per business) | `JOURNEY.md` §5; `results/run3_*` |
| 3 | + fine-tuned cross-encoder `intfloat/multilingual-e5-small` (MIT, 118M) as a stacked feature; bigger India retrieval + exact-name pass; 304,555 training businesses; 16 new features | `JOURNEY.md` §9; `results/run4_*` |
| 5 | + **dense retrieval** (fine-tuned `multilingual-e5-small` bi-encoder, top-15 neighbours, trained by L2 on Kaggle with validation businesses excluded) as a search pass + `rank_dense`/`cos_dense` features; + **2nd cross-encoder** `microsoft/mdeberta-v3-base` (MIT, 280M; L4, Kaggle) as `ce2_*` features. Retrieval recall 0.9688 → 0.9955; 8.3 candidates per business after pruning | `results/run6_*`, `kaggle/dense_retrieval.py`, `kaggle/mdeberta_cross_encoder.py` |
| 4 | + state/region-restricted name retrieval (`ber/regions.py`); exact-name pass for all countries; look-alike-digit / ID-tag name clean-up; wider e5 band; `candidate_pairs.tsv` trimmed to matcher p ≥ 0.0001 (22.6 → ~7 candidates per business) | `JOURNEY.md` §13; `results/run5_*` |

## How the exact code was pinned

- `dev_run3/`, `dev_run4/` and `submission/code/` were committed while those runs were current.
- `dev_run5/` was **reconstructed and verified byte-exact**. The first snapshot (commit `75ff884`)
  lacked one later, inert hook in `pipeline.py`, which merges `work_dir/ce2` only if that folder
  exists; it did not exist during run 5. The reconstruction was checked by applying the run-6 patch
  (`dev_run6/scripts/patch_run6.py`) to it: the result is identical, file for file, to the live code
  that run 6 uses.
- `dev_run6/` is the exact run-6 code (verified identical to the live code that produced submission 5).

## Verify a file

```bash
gunzip -c results/run5_matching_results_LB0.967.tsv.gz | sha256sum   # must equal row 4 above
```
