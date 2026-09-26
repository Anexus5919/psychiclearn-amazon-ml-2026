# PsychicLearn: Team Plan for the Final ~36 Hours (4 laptops in parallel)

Deadline: **27 Sep 2026, 23:59 IST**. Current public score: **0.941**; the top teams are at about 0.98.
Read `JOURNEY.md` (in the GitHub repo) first for the background. This file is the work plan.

---

## 0. How parallel work helps (the simple picture)

Think of a kitchen. **Laptop L1 is the head chef.** It holds the full dataset, the big intermediate
files (about 40 GB) and the main pipeline, and it makes every leaderboard upload. The other three laptops
each **prepare one ingredient** that L1 adds to the final dish.

Everyone works at the same time without clashing because:

- each ingredient is **independent**: it needs only small files that L1 exports, not the whole dataset;
- each person writes **only their own new files**. Nobody edits the same code file or data file;
- **L1 alone** merges the results, re-validates them and uploads. The Unstop rules allow only the team
  leader to submit, and forbid simultaneous logins.

### What depends on what

```
L1: run 4 (retrieval + pruning + new features + small cross-encoder) --> upload #3
     |                                                                      |
     |-- exports ce_train.parquet ----------> L4: big cross-encoder (mDeBERTa) --> ce2 scores --+
     |-- exports run-3/run-4 OOF file ------> L2: smarter decisions + thresholds --> params -----+
     |-- exports run-3/run-4 train features > L3: GBDT tuning + 2nd model --------> params/model +
                                                                                                 v
                              L1: run 5 = run 4 + all ingredients + France pseudo-labels --> upload #4, #5
```

**Rule for every ingredient:** it is only used if it **improves the validation F0.5** on L1. Validation
has predicted the leaderboard to within ~0.002, so we never spend uploads on guesses.

---

## 1. Who does what

| Laptop | Person | Role | GPU needed? | Main output |
|---|---|---|---|---|
| **L1** | Adarsh (team leader) | Head chef: run 4 → run 5, exports, merging, **all uploads**, final zip | yes (runs now) | submissions |
| **L2** | teammate (same specs as L1) | **Decision layer:** smarter thresholds and per-business set selection | no (CPU only) | `decide` code + tuned params |
| **L3** | teammate (same specs as L1) | **Model tuning:** LightGBM hyperparameters + a 2nd GBDT (CatBoost/XGBoost) for an ensemble | no (CPU) | params JSON + model file |
| **L4** | teammate (**best laptop**) | **Big cross-encoder:** fine-tune mDeBERTa-v3-base (on its own GPU if ≥ 12 GB VRAM, else Kaggle) | yes / Kaggle | cross-encoder scores |

Documentation for the final package is split: L2 writes the methodology first draft after their task;
L1 finalises it.

---

## 2. One-time setup (all teammates, ~30–45 min)

1. **GitHub.** Send Adarsh your GitHub username and accept the invite to the private repo
   `Anexus5919/psychiclearn-amazon-ml-2026`. Then:
   ```bash
   git clone https://github.com/Anexus5919/psychiclearn-amazon-ml-2026.git
   cd psychiclearn-amazon-ml-2026
   git checkout -b <your-branch>        # L2: feat/decision   L3: feat/gbdt   L4: feat/mdeberta
   ```
2. **Python 3.12 environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate               # Windows
   pip install -r submission/code/business_entity_resolution/requirements.txt
   ```
   - **L4 only (GPU):**
     `pip install torch --index-url https://download.pytorch.org/whl/cu128` then
     `pip install transformers==4.57.1 sentencepiece protobuf`.
     Check it with `python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"`.
3. **Shared files = GitHub Releases** (no Google Drive). Big data files cannot go into git itself (100 MB
   limit), so each person has one **release** in our repo, used as a file drop:
   `from-L1` (Adarsh), `from-L2`, `from-L3`, `from-L4`. Each file can be up to 2 GB, and it's free.
   - **Download:** repo page, then **Releases** (right sidebar), then e.g. `from-L1`, then click the file
     under *Assets*. Or with the GitHub CLI:
     `gh release download from-L1 -p "ce_train.parquet" -R Anexus5919/psychiclearn-amazon-ml-2026`
   - **Upload (only to your own release):** Releases, then your `from-Lx`, then the pencil icon (Edit),
     then drag the files into *Attach binaries*, then **Update release**. Or:
     `gh release upload from-L4 ce2_train.parquet ce2_test.parquet --clobber -R Anexus5919/psychiclearn-amazon-ml-2026`
   - Small files (code, params `.json`, notes) go on your git branch as usual.
   - Nobody needs the raw dataset except L1, but it is on the challenge portal if you want to look.
4. **Rules:**
   - Never log in to the Unstop account (simultaneous logins can disqualify us).
   - No external data or APIs.
   - Only MIT/Apache models, 8B parameters or fewer.

---

## 3. Task cards

### L2: decision layer (CPU, ~4–5 h of work)

**Why.** Our model's probabilities are turned into lists by a simple rule: "exclusive assignment + first
match ≥ t1 + later matches ≥ t2". The error analysis showed that **missed matches cost 1.6 points** and
**false merges cost 0.8**, so better decisions on the same probabilities can gain points for free.

**Inputs** (release `from-L1`):

| File | Columns | Size |
|---|---|---|
| `run3_oof.parquet` | `s1_id, cand_id, src (2/3), country, p, label` | ~3.1M rows (out-of-fold probabilities for validation businesses) |
| `run3_truth.parquet` | `s1_id, country, n_true, n_true_found` | one row per validation business |

`n_true_found` is how many true matches retrieval found.

**Steps:**
1. Re-implement the metric: `ber/scoring.py` has `f05_entity`. **Entities with no candidate rows still
   count:** empty predictions and missed pairs are false negatives. Check that you reproduce the current
   rule's score: `decide.tune_thresholds` gives about **0.9613**.
2. Try, in order of expected gain:
   - **(a)** separate `t1/t2` for S2 and S3 (coordinate search);
   - **(b)** a 3rd threshold for rank ≥ 3;
   - **(c)** **expected-F0.5 set selection**: for each business, sort its candidates by `p` and choose the
     top-k (k = 0…m) that maximises the expected F0.5, treating each candidate as correct with probability
     `p` (Monte-Carlo or exact enumeration for small m), plus the expected number of retrieval misses;
   - **(d)** the probability-sum rule: `k ≈ round(sum(p))`, capped by thresholds.
3. Guard against overfitting: tune on half of the businesses (e.g. `hash(s1_id) % 2 == 0`) and report the
   score on the other half.

**Output** (release `from-L2` + branch `feat/decision`):
- `decide.py` with a new function `decide_v2(df, params)`; unit tests keep passing;
- `decision_params.json`;
- a short note: old vs new F0.5 on the held-out half.

**Later:** when L1 posts `run4_oof.parquet`, re-run the tuning on it (~15 min) and post the updated params.
**Then:** draft `Documentation_template.md` from `JOURNEY.md`.

### L3: model tuning + second GBDT (CPU, ~4–6 h of compute)

**Why.** LightGBM runs with hand-picked settings. Tuning plus a different algorithm (CatBoost) usually
adds a few tenths of a point, and averaging two good models is safer than one.

**Input** (release `from-L1`): `run3_train_features.parquet`, about 3.1M rows. It has all feature columns plus
`s1_id`, `cand_id`, `label`, and the list of feature names in `features.json`.

**Steps:**
1. Use **4-fold grouped CV by `s1_id`**, with exactly the same folds as ours (`ber/model.py: fold_ids`,
   seed 42). Metric: **log-loss and AUC-PR** on out-of-fold data. L2's F0.5 tooling can score them too.
2. **LightGBM:** a small Optuna search (≈ 20–30 trials on a 30% sample of businesses) over:

   | Parameter | Range |
   |---|---|
   | `num_leaves` | 63–511 |
   | `learning_rate` | 0.02–0.1 |
   | `min_data_in_leaf` | 50–1000 |
   | `feature_fraction` | 0.5–1.0 |
   | `lambda_l2` | 0–10 |

   Then confirm the best on the full data.
3. **CatBoost** (Apache-2.0, `pip install catboost`): train with the same folds (CPU,
   `depth 8, lr 0.08, 2000 iters, early stopping`). Report its OOF log-loss and the log-loss of the average
   `0.5·LGB + 0.5·CB`.

**Output** (release `from-L3` + branch `feat/gbdt`):
- `lgb_params.json`;
- the CatBoost settings;
- an OOF comparison table;
- `oof_catboost.parquet` (`s1_id, cand_id, p_cb`), so L2 can test the averaged probabilities.

**Later:** when L1 posts `run4_train_features.parquet` (~ +2–3 h), repeat step 2's confirmation and step 3
on it. The run-4 features are the ones used in the final run.

### L4: big cross-encoder (GPU / Kaggle, ~3–4 h wall, mostly unattended)

**Why.** A fine-tuned multilingual transformer reads both records side by side and learns typos, dropped
digits, duplicated words and transliteration. Our small model (e5-small) is training on L1 now. The
**stronger mDeBERTa-v3-base (MIT, 280M)** needs more GPU memory than L1 has.

**Inputs** (release `from-L1`):

| File | Columns | When |
|---|---|---|
| `ce_train.parquet` | `text_a, text_b, label` | now: ~1.0M labelled pairs from businesses disjoint from everything else |
| `score_train.parquet`, `score_test.parquet` | `s1_id, cand_id, text_a, text_b` | later: the uncertain pairs to score |

**Where to run.**
- Your GPU **≥ 12 GB**: run locally.
- Otherwise: **Kaggle** (free, 2×T4, runs in the background).

**Script:** `kaggle/mdeberta_cross_encoder.py` in the repo.
- **Local:**
  ```bash
  set BER_IN=C:\path\to\downloaded_files
  set BER_OUT=C:\path\to\out
  python kaggle/mdeberta_cross_encoder.py
  ```
- **Kaggle:** follow the step-by-step guide `kaggle/KAGGLE_GUIDE_L4.md`. It covers setup, a 5-minute smoke
  test, the training run, the score-only run and troubleshooting.

**Phases:**
- **Now:** train on `ce_train.parquet` (estimate 1–2 h on 2×T4). The script saves the model to `mdeberta_ce/`.
- **When L1 posts the score files:** attach the phase-1 notebook output as an input. The script sees
  `mdeberta_ce/`, switches to SCORE-ONLY mode and writes `ce2_train.parquet` / `ce2_test.parquet`
  (estimate 1.5–3 h: 15.2M pairs).

**Output** (release `from-L4`): `ce2_train.parquet` and `ce2_test.parquet` (`s1_id, cand_id, ce_p`), plus a note
of the training time and final loss.

### L1: head chef (Adarsh's laptop, with Claude)

| When | What |
|---|---|
| now | finish **run 4**: India retrieval → pruning → new features → e5-small cross-encoder scores → final LightGBM → **upload #3** |
| now (+45 min) | export for teammates: `ce_train.parquet`, `run3_oof.parquet` + `run3_truth.parquet`, `run3_train_features.parquet` |
| after run-4 pruning | export `score_train.parquet` / `score_test.parquet` for L4 |
| after run 4 | export `run4_oof.parquet`, `run4_train_features.parquet` for L2/L3 |
| run 5 | merge: L4's `ce2` scores as a feature + L3's params (and CatBoost if it helps) + L2's decision rule + **France pseudo-labelling** → validate → **upload #4** |
| 27 Sep | final tuning run → **upload #5 (the best-validated file)**; final zip + documentation |

---

## 4. Timeline (T0 = when this plan is shared)

| Time | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| T0 → T0+0.75h | run 4 continues; exports posted | setup | setup | setup + Kaggle dataset upload |
| T0+0.75 → T0+3h | run 4 → upload #3 | decision work on run-3 OOF | tuning on run-3 features | **mDeBERTa training** |
| T0+2.5h | posts `score_*.parquet` | ↓ | ↓ | scores the uncertain pairs |
| T0+3.5h | posts run-4 OOF + features | re-tune on run 4 | confirm on run 4 | posts `ce2_*` |
| T0+4.5 → T0+6h | **run 5** with everything → upload #4 | documentation draft | – | – |
| 27 Sep | final run + upload #5 + zip | review docs | review | – |

---

## 5. Clash-proof rules

1. **Code:** each person works on their own branch and in their own files. L1 merges through pull requests.
2. **Data:** never edit another person's file. Write new files into your own `from-Lx` release, named
   exactly as in the task cards.
3. **Same folds everywhere:** `fold_ids(s1_id, 4, seed 42)`. Otherwise validation numbers aren't comparable.
4. **Same metric everywhere:** `ber/scoring.py`. Every claim of improvement comes with **old vs new F0.5
   on held-out businesses**.
5. **Only L1 uploads to Unstop,** and the **last upload must be the best-validated file**.
6. **No real money:** free Kaggle or Colab tiers only, no paid cloud plans.

## 6. If something goes wrong

| Problem | Fallback |
|---|---|
| Kaggle GPU quota is used up | Train mDeBERTa for ½ epoch, or use XLM-R-base; if L4 is out entirely, L1 keeps e5-small |
| A teammate's result doesn't beat validation | It isn't used. No harm done. |
| Run 4 fails on L1 | Run 3's file (LB 0.941) is always kept as a safe fallback |
| The deadline is near | Stop experiments by **27 Sep 20:00 IST**; final upload and zip by 23:00 |
