#!/usr/bin/env bash
# Run-4 CPU chain: waits for the cross-encoder data prep (python.exe running ber.ce_data) to finish,
# then retrieval (US train; India train+test with bigger budgets + exact-name pass), pruning, augment.
S="/c/Users/ANEXUS/AppData/Local/Temp/claude/c--Users-ANEXUS-Downloads-6ab10eb3b23ba-student-resource/77518dc9-b0d4-484a-b577-5d70f9aadbd6/scratchpad"
PY="$S/berenv/Scripts/python.exe"
DATA="C:/Users/ANEXUS/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset"
W4="C:/Users/ANEXUS/Downloads/PsychicLearn_work4"
OUT="C:/Users/ANEXUS/Downloads/PsychicLearn_run4_output"
# wait only on python.exe processes (this bash script's own command line can never match)
while powershell -NoProfile -c "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*ber.ce_data*' }) { exit 0 } else { exit 1 }"; do sleep 60; done
cd "/c/Users/ANEXUS/Downloads/PsychicLearn_dev/src" || exit 1
run() { PYTHONIOENCODING=utf-8 "$PY" -m ber.pipeline --data-dir "$DATA" --work-dir "$W4" --out-dir "$OUT" "$@" >> "$W4/run4_cpu.txt" 2>&1 || { echo "FAILED: $*"; tail -n 8 "$W4/run4_cpu.txt"; exit 1; }; }
run --stage pairs --only-countries US
run --stage pairs --only-countries India --k-name 15 --k-addr 15 --k-combo 20 --exact-name-cap 50
run --stage prune
run --stage augment
echo "RUN4 CPU CHAIN DONE"
grep -E "pass exact|pre-ranker rule|pruned (train|test):|augment done" "$W4/log.txt" | cut -c1-300
