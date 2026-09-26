#!/usr/bin/env bash
# Run 5: region-restricted name pass + exact pass everywhere + name clean-up, wider e5 band.
# pairs (India, then US+France) -> prune -> e5 band/todo -> [e5 scoring on GPU || augment on CPU] -> merge
# -> augment_ce -> train -> predict -> official validator.
S="/c/Users/ANEXUS/AppData/Local/Temp/claude/c--Users-ANEXUS-Downloads-6ab10eb3b23ba-student-resource/77518dc9-b0d4-484a-b577-5d70f9aadbd6/scratchpad"
SW="C:/Users/ANEXUS/AppData/Local/Temp/claude/c--Users-ANEXUS-Downloads-6ab10eb3b23ba-student-resource/77518dc9-b0d4-484a-b577-5d70f9aadbd6/scratchpad"
PY="$S/berenv/Scripts/python.exe"
DATA="C:/Users/ANEXUS/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset"
W5="C:/Users/ANEXUS/Downloads/PsychicLearn_work5"
OUT="C:/Users/ANEXUS/Downloads/PsychicLearn_run5_output"
E5="C:/Users/ANEXUS/Downloads/PsychicLearn_work_ce/ce_e5small"
LOG="$W5/run5_chain.txt"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd "/c/Users/ANEXUS/Downloads/PsychicLearn_dev/src" || exit 1
fail() { echo "FAILED: $* $(date +%H:%M)"; tail -n 12 "$LOG"; exit 1; }
run() { echo "== $* $(date +%H:%M)"; "$PY" -m ber.pipeline --data-dir "$DATA" --work-dir "$W5" --out-dir "$OUT" --exact-name-cap 50 --k-region 10 "$@" >> "$LOG" 2>&1 || fail "$*"; }
run --stage pairs --only-countries India --k-name 15 --k-addr 15 --k-combo 20
run --stage pairs --only-countries US,France
grep -E "pass region|pass exact|pairs written" "$W5/log.txt" | tail -n 30 | cut -c1-160
run --stage prune
grep -E "pre-ranker rule|pruned (train|test):" "$W5/log.txt" | tail -n 3 | cut -c1-200
echo "== e5 band $(date +%H:%M)"
"$PY" "$SW/ce_band5.py" band "$W5" 0.005 0.995 >> "$LOG" 2>&1 || fail "band"
tail -n 2 "$LOG"
( for sp in train test; do
    "$PY" -m ber.cross_encoder score --data-dir "$DATA" --split $sp --model-dir "$E5" \
      --pairs "$W5/ce/todo_$sp.parquet" --out "$W5/ce/scored_$sp.parquet" >> "$W5/run5_e5.txt" 2>&1 || exit 1
  done ) &
E5PID=$!
run --stage augment
echo "== waiting for e5 scoring $(date +%H:%M)"
wait $E5PID || fail "e5 scoring (see run5_e5.txt)"
"$PY" "$SW/ce_band5.py" merge "$W5" >> "$LOG" 2>&1 || fail "merge"
tail -n 2 "$LOG"
run --stage augment_ce --ce-dir "$W5/ce"
run --stage train
grep -E "fold [0-3]:|OOF macro|F0.5=" "$W5/log.txt" | tail -n 7 | cut -c1-200
run --stage predict
echo "== validator $(date +%H:%M)"
"$PY" "$DATA/../utils/validate_submission.py" --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" \
  --test-dir "$DATA/test" --check-ids 2>&1 | tail -n 8
echo "RUN5 DONE $(date +%H:%M)"
