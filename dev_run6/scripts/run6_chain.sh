#!/usr/bin/env bash
# Run 6 = run 5 + dense-retrieval pass (L2's fine-tuned e5 bi-encoder neighbours) [+ ce2 if work6/ce2 exists].
# Start ONLY after run 5 has finished and the dense gate passed. Usage: bash run6_chain.sh <dense_dir>
DENSE="$1"
S="/c/Users/ANEXUS/AppData/Local/Temp/claude/c--Users-ANEXUS-Downloads-6ab10eb3b23ba-student-resource/77518dc9-b0d4-484a-b577-5d70f9aadbd6/scratchpad"
SW="C:/Users/ANEXUS/AppData/Local/Temp/claude/c--Users-ANEXUS-Downloads-6ab10eb3b23ba-student-resource/77518dc9-b0d4-484a-b577-5d70f9aadbd6/scratchpad"
PY="$S/berenv/Scripts/python.exe"
DATA="C:/Users/ANEXUS/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset"
W5="/c/Users/ANEXUS/Downloads/PsychicLearn_work5"
W6="C:/Users/ANEXUS/Downloads/PsychicLearn_work6"
OUT="C:/Users/ANEXUS/Downloads/PsychicLearn_run6_output"
E5="C:/Users/ANEXUS/Downloads/PsychicLearn_work_ce/ce_e5small"
SRC="/c/Users/ANEXUS/Downloads/PsychicLearn_dev/src"
LOG="$W6/run6_chain.txt"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
[ -f "$DENSE/dense_train.parquet" ] && [ -f "$DENSE/dense_test.parquet" ] || { echo "dense files missing in $DENSE"; exit 1; }
mkdir -p "/c/Users/ANEXUS/Downloads/PsychicLearn_work6/norm" "/c/Users/ANEXUS/Downloads/PsychicLearn_work6/pairs/train" "/c/Users/ANEXUS/Downloads/PsychicLearn_work6/pairs/test"
cp -n "$W5"/norm/* "/c/Users/ANEXUS/Downloads/PsychicLearn_work6/norm/"
cp -n "$W5/pairs/train/_queries.parquet" "/c/Users/ANEXUS/Downloads/PsychicLearn_work6/pairs/train/"
cp -n "$W5/pairs/test/_queries.parquet" "/c/Users/ANEXUS/Downloads/PsychicLearn_work6/pairs/test/"
grep -q "DENSE_PASS" "$SRC/ber/blocking.py" || "$PY" "$SW/patch_run6.py" "$SRC" || { echo "patch failed"; exit 1; }
cd "$SRC" || exit 1
fail() { echo "FAILED: $* $(date +%H:%M)"; tail -n 12 "$LOG"; exit 1; }
run() { echo "== $* $(date +%H:%M)"; "$PY" -m ber.pipeline --data-dir "$DATA" --work-dir "$W6" --out-dir "$OUT" --exact-name-cap 50 --k-region 10 --dense-dir "$DENSE" "$@" >> "$LOG" 2>&1 || fail "$*"; }
run --stage pairs --only-countries India --k-name 15 --k-addr 15 --k-combo 20
run --stage pairs --only-countries US,France
run --stage prune
grep -E "pre-ranker rule|pruned (train|test):" "$W6/log.txt" | tail -n 3 | cut -c1-200
"$PY" "$SW/ce_band6.py" band "$W6" 0.005 0.995 >> "$LOG" 2>&1 || fail "band"
tail -n 2 "$LOG"
( for sp in train test; do
    "$PY" -m ber.cross_encoder score --data-dir "$DATA" --split $sp --model-dir "$E5" \
      --pairs "$W6/ce/todo_$sp.parquet" --out "$W6/ce/scored_$sp.parquet" >> "$W6/run6_e5.txt" 2>&1 || exit 1
  done ) &
E5PID=$!
run --stage augment
wait $E5PID || fail "e5 scoring (see run6_e5.txt)"
"$PY" "$SW/ce_band6.py" merge "$W6" >> "$LOG" 2>&1 || fail "merge"
run --stage augment_ce --ce-dir "$W6/ce"
run --stage train
grep -E "OOF macro|F0.5=" "$W6/log.txt" | tail -n 3 | cut -c1-200
run --stage predict
"$PY" "$DATA/../utils/validate_submission.py" --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" \
  --test-dir "$DATA/test" --check-ids 2>&1 | tail -n 6
echo "RUN6 DONE $(date +%H:%M)"
