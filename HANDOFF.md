# PsychicLearn — handoff (Amazon ML Challenge 2026, deadline 27 Sep 23:59 IST)

## Where things are
| Item | Path |
|---|---|
| Dataset (read-only, keep untouched) | `Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\` |
| Blueprint + EDA | `Downloads\amazon_ml_2026_analysis\` |
| **Submission package (final zip source)** | `Downloads\PsychicLearn_submission\` (code = run-2 version; `output\` = run-2 files, LB 0.931) |
| **Run-3 code (newer: transliteration + pruning)** | `Downloads\PsychicLearn_dev\src\` (copy into the package once run 3 is validated) |
| Run-2 work dir | `Downloads\PsychicLearn_work\` |
| Run-3 work dir / outputs | `Downloads\PsychicLearn_work3\` / `Downloads\PsychicLearn_run3_output\` |
| Python env (pinned reqs) | `...\scratchpad\berenv\Scripts\python.exe` (uv venv, py3.12) |

## Scores so far
- Run 2: OOF F0.5 0.9515 (US 0.9727, India 0.9196); **public LB 0.931** (submitted 25 Sep 23:59). France inferred ≈0.86–0.90 (weakest).
- LB top ≈ 0.988 (26 Sep ~00:30).

## Run 3 (started 26 Sep ~00:20)
Stages: prepare (learned translit, leak-free) → pairs (India re-blocked; US/France reused from run 2) → prune (pre-ranker keeps ~top-N per S1 with ≤0.2% recall loss; rule + frontier in `work3\prune\rule.json` and log) → train → predict.
Resume / finish any unfinished stages (each stage skips finished partitions):
```
cd Downloads\PsychicLearn_dev\src
<berenv python> -m ber.pipeline --data-dir <dataset> --work-dir C:\Users\ANEXUS\Downloads\PsychicLearn_work3 --out-dir C:\Users\ANEXUS\Downloads\PsychicLearn_run3_output --stage <prune|train|predict>
```
Then validate: from `student_resource\`: `python utils/validate_submission.py --matching <run3_output>\matching_results.tsv --candidate NONE --test-dir dataset/test --check-ids`

## Next ideas (in order)
1. Inspect France predictions (empty-rate, sample matches) → fix over/under-matching.
2. Choose pruning budget from the frontier (smaller candidate sets rank higher per organisers' update).
3. Bigger retrieval budgets / reverse pass for India recall; decision tuning.
4. Final: copy best code into package, fill `Documentation_template.md`, build `PsychicLearn_submission.zip` (no __MACOSX/.DS_Store), make the LAST leaderboard upload the best-validated file.

## Rules to remember
No real money on AWS (Free plan only; never upgrade/join Organizations). Heavy compute runs on this laptop. No external data/APIs. Output files LF-only.
