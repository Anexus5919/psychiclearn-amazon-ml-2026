# PsychicLearn — handoff (Amazon ML Challenge 2026, deadline 27 Sep 23:59 IST)

*Updated 26 Sep 2026, ~17:00 IST. Full story: `JOURNEY.md`. Work plan: `TEAM_PLAN.md` (§7 = current plan).*

## Scores so far
| Upload | Run | Validation F0.5 (OOF) | Public LB |
|---|---|---|---|
| #1 (25 Sep 23:59) | Run 2 | 0.9515 (US 0.9727, India 0.9196) | 0.931 |
| #2 (26 Sep 02:50) | Run 3 | 0.9613 (US 0.9726, India 0.9443) | 0.941 |
| **#3 (26 Sep 15:44)** | **Run 4** | **0.97628** (US 0.9824, India 0.9671; t1 0.54, t2 0.74) | **0.963 (current best)** |

- France (no labels) worked out from the LB: ≈0.85 (run 3) → ≈0.90 (run 4).
- LB top 0.9906 (26 Sep afternoon). Uploads used: 1 on 25 Sep, 2 on 26 Sep (limit 5/day).

## Where things are (L1 laptop)
| Item | Path |
|---|---|
| Dataset (read-only, keep untouched) | `Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\` |
| Newest code | `Downloads\PsychicLearn_dev\src\` (repo: `dev_run5/`) |
| Run-4 work dir / outputs (best so far) | `Downloads\PsychicLearn_work4\` / `Downloads\PsychicLearn_run4_output\` |
| Run-5 work dir / outputs (running) | `Downloads\PsychicLearn_work5\` / `Downloads\PsychicLearn_run5_output\` |
| e5 cross-encoder model | `Downloads\PsychicLearn_work_ce\ce_e5small\` |
| e5 scores (Kaggle) | `Downloads\PsychicLearn_work4\ce\{train,test}.parquet` |
| Files shared with the team | GitHub release `from-L1` (staged in `Downloads\PsychicLearn_share\from_L1\`) |
| Submission package (final zip source) | `Downloads\PsychicLearn_submission\` (still run-2 code: update on 27 Sep) |

## Run 5 (running since 26 Sep 16:32; ETA ~21:00)
`run5_chain.sh` (copy in `dev_run5/scripts/`): pairs (India k 15/15/20; US/France 10/10/15; exact-name cap 50; **region pass k 10**) → prune → e5 band [0.005, 0.995] (run-4 scores reused, new pairs scored on the GPU while `augment` runs) → augment_ce → train → predict (`candidate_pairs.tsv` keeps p ≥ 0.0001) → validator.
Resume one stage: `python -m ber.pipeline --data-dir <dataset> --work-dir <work5> --out-dir <run5_output> --exact-name-cap 50 --k-region 10 --stage <stage>`.
**Upload only if OOF > 0.97628.**

## Decided
- France pseudo-labelling: **rejected** (proxy test: −0.0012 F0.5).
- L2 threshold variants: +0.00004 (noise) → not merged.
- Candidate trimming p ≥ 0.0001: 22.6 → 7.3 candidates/S1 for a 0.003% recall loss → used from run 5.

## Next
1. Run 5 → validator → upload #4 if better.
2. L4 `ce2_*` (mDeBERTa) → `--stage augment_ce --ce-dir <dir> --ce-prefix ce2` → train → predict → upload if OOF gain ≥ +0.0005.
3. 27 Sep: copy the best code into the package; **fix the package README** (it still says "no pretrained models, CPU only"); documentation; `PsychicLearn_submission.zip`; last upload = best-validated file. Experiments stop 20:00 IST.

## Rules to remember
No real money (AWS Free plan only; free Kaggle only). No external data/APIs; pretrained models only MIT/Apache ≤ 8B. Output files LF-only; validator with `--check-ids` before every upload. Only Adarsh uploads.

## Final state (28 Sep 2026)

- **Best submission:** #5 = run 6, public LB **0.982608** (validation 0.98769). The leaderboard closed on
  27 Sep 23:59 IST.
- **Code Submission round (until 29 Sep 10:00 IST):** upload `Downloads\PsychicLearn_submission.zip`.
  It contains the run-6 outputs (byte-identical to the leaderboard upload), the pipeline code, and the
  filled documentation. Its content is in this repo under `submission/`; checksums are in
  `SUBMISSIONS.md`.
