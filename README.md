# PsychicLearn: Amazon ML Challenge 2026 (Business Entity Resolution)

Private team repository. For each Source-1 business record, the task is to find the Source-2/3 records of the same business. The metric is macro F0.5.

## Status

| Run | What changed | Validation F0.5 (held-out train) | Public LB |
|---|---|---|---|
| Run 2 | Baseline: word/char TF-IDF retrieval → LightGBM → exclusive assignment + rank thresholds | 0.9515 (US 0.973, India 0.920) | **0.931** |
| Run 3 | + learned Indic→Latin transliteration (India), + learned pruning (54 → 18 candidates per S1) | 0.9613 (US 0.973, India 0.944) | **0.941** |
| Run 4 | + e5-small cross-encoder (MIT, 118M; stacked as a feature), bigger India search + exact-name pass, 304k training businesses, 16 new features | **0.9763** (US 0.982, India 0.967) | **0.963** |
| Run 5 | + state/region-restricted name search, exact-name pass everywhere, look-alike/ID-tag name clean-up, wider e5 band, candidate file 22.6 → ~7 per S1 | 0.9779 (US 0.984, India 0.969) | **0.967** |
| Run 6 | + dense retrieval (fine-tuned e5 bi-encoder, L2) as a search pass + features; + mDeBERTa-v3-base cross-encoder (L4) | running (26 Sep 23:31 →) | – |

France has no training labels. Worked out from the leaderboard, it scores ≈0.85 (run 3) and ≈0.90 (run 4). Full story and metrics: `JOURNEY.md`. Plan: `TEAM_PLAN.md` §7.

## Layout

| Folder | Contents |
|---|---|
| `submission/` | Final-package structure. `code/business_entity_resolution/` is the version that produced the LB 0.931 file. `output/` stays empty here: the data files are too big for git. |
| `dev_run3/` | Run-3 code snapshot |
| `dev_run4/` | Run-4 code snapshot (cross-encoder, extra/group features, exports) |
| `dev_run5/` | **Newest code** (run 5: `ber/regions.py`, region pass in `ber/blocking.py`, `--ce-prefix ce2`, candidate trimming) + `scripts/` (run chain, error analyses, pseudo-label proxy test). It moves into `submission/` once validated. |
| `analysis/` | `ER_Solution_Blueprint.md` (full problem analysis and plan) plus the EDA scripts and their outputs |
| `results/` | Gzipped `matching_results.tsv` of runs 2–4, validation reports and pruning rules, run-4 error analysis, pseudo-label proxy test, name-crowding measurement |
| `kaggle/` | Cross-encoder script for Kaggle (train + score, or score-only with an attached model) and `KAGGLE_GUIDE_L4.md` |
| Releases `from-L1` … `from-L4` | Big data files shared between teammates (validation data, features, cross-encoder inputs/outputs) |
| `SUBMISSIONS.md` | **Version history of every leaderboard upload** (file SHA-256, code folder, git tag, scores) |
| `dev_run6/` | Code of the next run (dense-retrieval pass + 2nd cross-encoder) + its scripts |
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

- No external data or APIs. Everything is learned from the provided files.
- Pretrained models only if MIT/Apache and ≤ 8B parameters. Used: `intfloat/multilingual-e5-small` (MIT, 118M) from run 4 onwards; `microsoft/mdeberta-v3-base` (MIT, 280M) if it passes validation. Both are fine-tuned on the provided training data only.
- Output TSVs use LF line endings. Run `utils/validate_submission.py --check-ids` before every upload.
- The last leaderboard upload must be our best-validated file.
