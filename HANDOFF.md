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
- **Run 3: OOF 0.9613 (US 0.9726, India 0.9443); 18.0 candidates per S1 (pruned from 53.5; rule top-30 & p≥0.001, 0.18% recall loss); public LB 0.941** (submitted 26 Sep 02:50). Output: `Downloads\PsychicLearn_run3_output\` (validator PASS). This is currently our best file.
- Pruning frontier: 0.5% loss → 11.9 cand/S1; 1% loss → 10.0 cand/S1.
- LB top ≈ 0.988 (26 Sep ~00:30). Uploads used: 1 on 25th, 1 on 26th (limit 5/day).

## Run 3 (COMPLETE: started 26 Sep ~00:20, finished 02:47)
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
