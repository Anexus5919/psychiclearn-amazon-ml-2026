#!/usr/bin/env bash
# Run-4 second half: e5 scoring of the uncertain band (GPU, overlaps the CPU augment stage), then
# augment_ce -> train -> predict. Train/France bands are written before this starts (files are final).
S="/c/Users/ANEXUS/AppData/Local/Temp/claude/c--Users-ANEXUS-Downloads-6ab10eb3b23ba-student-resource/77518dc9-b0d4-484a-b577-5d70f9aadbd6/scratchpad"
PY="$S/berenv/Scripts/python.exe"
DATA="C:/Users/ANEXUS/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset"
W4="C:/Users/ANEXUS/Downloads/PsychicLearn_work4"
CE="$W4/ce"
MODEL="C:/Users/ANEXUS/Downloads/PsychicLearn_work_ce/ce_e5small"
OUT="C:/Users/ANEXUS/Downloads/PsychicLearn_run4_output"
LOG="$W4/run4_ce.txt"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
fail() { echo "FAILED: $*"; tail -n 8 "$LOG"; exit 1; }
# 1) test India/US bands as soon as pruning is finished (augment rewrites test India only ~10+ min later)
until grep -q "pruned test:" "$W4/log.txt"; do sleep 60; done
"$PY" "$S/ce_band.py" test India US >> "$LOG" 2>&1 || fail "band test"
# 2) score every band file on the GPU
cd "/c/Users/ANEXUS/Downloads/PsychicLearn_dev/src" || exit 1
for sc in train_India train_US test_France test_India test_US; do
  split="${sc%%_*}"
  echo "== scoring $sc $(date +%H:%M)" >> "$LOG"
  "$PY" -m ber.cross_encoder score --data-dir "$DATA" --split "$split" --model-dir "$MODEL" \
      --pairs "$CE/band_$sc.parquet" --out "$CE/ce_$sc.parquet" >> "$LOG" 2>&1 || fail "score $sc"
done
"$PY" -c "
import glob, pandas as pd
for s in ('train', 'test'):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(r'$CE/ce_' + s + '_*.parquet'))], ignore_index=True)
    df[['s1_id', 'cand_id', 'ce_p']].to_parquet(r'$CE/' + s + '.parquet', index=False)
    print(s, len(df), 'ce_p mean', round(float(df.ce_p.mean()), 4))
" >> "$LOG" 2>&1 || fail "concat"
echo "CE SCORING DONE $(date +%H:%M)"
# 3) wait for the CPU chain (augment) to finish, then the rest of run 4
until grep -q "augment done" "$W4/log.txt"; do sleep 60; done
run() { "$PY" -m ber.pipeline --data-dir "$DATA" --work-dir "$W4" --out-dir "$OUT" "$@" >> "$W4/run4_cpu.txt" 2>&1 || fail "$*"; }
run --stage augment_ce --ce-dir "$CE"
echo "augment_ce done $(date +%H:%M)"
run --stage train
grep -E "OOF macro|F0.5=" "$W4/log.txt" | tail -n 4
run --stage predict
echo "RUN4 DONE $(date +%H:%M)"
tail -n 5 "$W4/log.txt" | cut -c1-200
