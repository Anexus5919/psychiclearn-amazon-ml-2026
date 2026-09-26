#!/usr/bin/env bash
# Run-4 final: merge e5 scores (Kaggle) -> train (4-fold OOF + thresholds) -> predict -> official validator.
S="/c/Users/ANEXUS/AppData/Local/Temp/claude/c--Users-ANEXUS-Downloads-6ab10eb3b23ba-student-resource/77518dc9-b0d4-484a-b577-5d70f9aadbd6/scratchpad"
PY="$S/berenv/Scripts/python.exe"
DATA="C:/Users/ANEXUS/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset"
W4="C:/Users/ANEXUS/Downloads/PsychicLearn_work4"
OUT="C:/Users/ANEXUS/Downloads/PsychicLearn_run4_output"
export PYTHONIOENCODING=utf-8
cd "/c/Users/ANEXUS/Downloads/PsychicLearn_dev/src" || exit 1
run() { echo "== $* $(date +%H:%M)"; "$PY" -m ber.pipeline --data-dir "$DATA" --work-dir "$W4" --out-dir "$OUT" "$@" >> "$W4/run4_final.txt" 2>&1 || { echo "FAILED: $*"; tail -n 12 "$W4/run4_final.txt"; exit 1; }; }
run --stage augment_ce --ce-dir "$W4/ce"
run --stage train
grep -E "train pairs:|fold [0-3]:|OOF macro|F0.5=" "$W4/log.txt" | tail -n 9 | cut -c1-200
run --stage predict
echo "== validator $(date +%H:%M)"
"$PY" "C:/Users/ANEXUS/Downloads/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py" \
  --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" --test-dir "$DATA/test" --check-ids 2>&1 | tail -n 15
echo "RUN4 FINAL DONE $(date +%H:%M)"
