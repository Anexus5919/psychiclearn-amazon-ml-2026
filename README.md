# PsychicLearn: Amazon ML Challenge 2026 (Business Entity Resolution)

Private team repository. For each Source-1 business record, the task is to find the Source-2/3 records of the same business. The metric is macro F0.5.

## Status

| Run | What changed | Validation F0.5 (held-out train) | Public LB |
|---|---|---|---|
| Run 2 | Baseline: word/char TF-IDF retrieval → LightGBM → exclusive assignment + rank thresholds | 0.9515 (US 0.973, India 0.920) | **0.931** |
| Run 3 | + learned Indic→Latin transliteration (India), + learned pruning (54 → 18 candidates per S1) | 0.9613 (US 0.973, India 0.944) | **0.941** |

## Layout

| Folder | Contents |
|---|---|
| `submission/` | Final-package structure. `code/business_entity_resolution/` is the version that produced the LB 0.931 file. `output/` stays empty here: the data files are too big for git. |
| `dev_run3/` | Newest code: transliteration, learned pruning, cross-encoder (`ber/cross_encoder.py`), run-4 extra/group features (`ber/features_extra.py`), exports (`ber/ce_export.py`). It moves into `submission/` once validated. |
| `analysis/` | `ER_Solution_Blueprint.md` (full problem analysis and plan) plus the EDA scripts and their outputs |
| `results/` | Gzipped `matching_results.tsv` of runs 2 and 3, plus run 3's validation report and pruning rule |
| `kaggle/` | mDeBERTa cross-encoder script (Kaggle or any GPU) |
| `TEAM_PLAN.md` | **Who does what, in parallel, until the deadline** |
| `HANDOFF.md` | Current state, paths and next steps |

## Running

See `submission/code/business_entity_resolution/README.md`. You need Python 3.12 and `pip install -r requirements.txt`, then:

```
cd src
python -m ber.pipeline --data-dir <dataset> --work-dir <scratch> --out-dir <output>
```

**The dataset is not in this repo.** Download it from the challenge portal.

## Rules we follow

- No external data, APIs or pretrained models; everything is learned from the provided files.
- Output TSVs use LF line endings. Run `utils/validate_submission.py --check-ids` before every upload.
- The last leaderboard upload must be our best-validated file.
