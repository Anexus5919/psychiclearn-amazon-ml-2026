# Business Entity Resolution: team PsychicLearn

Amazon ML Challenge 2026. This folder regenerates `output/matching_results.tsv` and
`output/candidate_pairs.tsv` from the provided `train/` and `test/` files.

- **Result:** our best submission is run 6. Public leaderboard macro F0.5 is **0.982608**. Validation
  (out-of-fold, 304,555 held-out training businesses) is **0.98769**: India 0.98772, US 0.98767.
- **Outputs submitted** (`../../output/`, LF line endings, validated with
  `utils/validate_submission.py --check-ids`):
  - `matching_results.tsv`: SHA-256 `3f1e43a2b5cc52f7354efbd83efd47ed9b4049b82600a5465d8586cf85512e0c`,
    the file uploaded to the leaderboard. 1,732,544 rows; 5,784,246 matched IDs (3.34 per business);
    100,477 empty.
  - `candidate_pairs.tsv`: SHA-256 `df87021974f5bd9f269372dd69cef222b3e94b96a13f565b761728dd72e79554`.
    11,008,665 candidate IDs (6.35 per business).
- **No external data or APIs.** Every rule and model is learned from the provided files. Pretrained
  models (downloaded from Hugging Face) are all MIT-licensed and far below the 8B-parameter limit:
  - `intfloat/multilingual-e5-small` (118M), fine-tuned as a cross-encoder and as a dense bi-encoder;
  - `microsoft/mdeberta-v3-base` (280M), fine-tuned as a second cross-encoder.

## 1. Setup

```bash
cd code/business_entity_resolution
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA build for the GPU steps
pip install -r requirements.txt
python -m pytest -q src/tests            # unit tests: metric, normalisers, decision rule, writer
```

The dataset folder must contain `train/` and `test/` exactly as distributed (`.tsv` or `.tsv.gz`).

## 2. Reproduce both output files (one command)

```bash
cd src
python run_all.py --data-dir <path>/dataset --work-root <scratch dir> --out-dir <path>/output \
                  --validator <path>/utils/validate_submission.py
```

- **Resumable:** every step writes to `--work-root` and skips finished work.
- **Selected steps only:** `--steps pairs,prune,...` runs just those steps.
- **Quick check:** first build a small copy of the data with
  `python tests/make_dev_subset.py --data-dir <dataset> --out-dir <small dataset>`, then run
  `python run_all.py --data-dir <small dataset> ... --smoke`. This runs **every** step end to end in
  about 20 minutes on a laptop GPU. We ran this check on this exact package before
  submitting: all 13 steps completed and the official validator passed.

| Step | What it does | Hardware / time (full data) |
|---|---|---|
| `base` | normalise all 12M records (learns the Indic→Latin transliteration dictionary); base retrieval + learned pruning on the base training businesses | CPU, ~1.5 h |
| `ce_data` | candidate pairs for a separate 3% of businesses: cross-encoder training data | CPU, ~20 min |
| `ce_train` | fine-tune `multilingual-e5-small` as a cross-encoder (1.02M pairs, 1 epoch, batch 128, lr 6e-5) | GPU, ~55 min on RTX 3050 6 GB |
| `queries` | final 304,555 training/validation businesses (disjoint from `ce_data`) | CPU, seconds |
| `dense` | fine-tune `multilingual-e5-small` as a bi-encoder (2M positive pairs, in-batch negatives), top-15 neighbours for every S1 | GPU, ~2 h on Kaggle T4 ×2 |
| `regions` | state/region key of every record (S1 vocabulary + map learned from training pairs) | CPU, ~10 min |
| `pairs` | retrieval + ~110 pair features | CPU, ~2.5 h (8 threads) |
| `prune` | learned pre-ranker keeps ~8.3 candidates per business (≤0.2% true-match loss) | CPU, ~20 min |
| `ce_score` | e5 cross-encoder score for pairs with pre-ranker p ∈ [0.005, 0.995] (~15M) | GPU, ~1.5 h |
| `mdeberta` | fine-tune `mdeberta-v3-base` cross-encoder on the `ce_data` pairs, then score pairs with p ∈ [0.02, 0.995] | GPU, ~7.5 h on Kaggle T4 ×2 |
| `augment` | extra/group features, merge both cross-encoders' scores | CPU, ~15 min |
| `train` | LightGBM, 4-fold CV grouped by business; thresholds tuned for macro F0.5 | CPU, ~10 min |
| `predict` | score test pairs, exclusive assignment + thresholds, write and self-check both TSVs | CPU, ~10 min |

Peak RAM is about 12 GB (we used a 16 GB laptop). The GPU scripts in `src/gpu/` also run unchanged as
Kaggle notebooks: attach the inputs as a dataset; they search `/kaggle/input`.

## 3. How the pipeline works

1. **Normalise** (`ber/normalize.py`, `ber/translit.py`).
   - Names and addresses are lower-cased and accent-folded (Latin letters only).
   - DBA / handle forms and legal forms are canonicalised; street types and French abbreviations are
     expanded; house numbers are parsed.
   - Indic-script tokens are converted to Latin with a dictionary **learned from training pairs**
     (validation businesses excluded).
2. **Retrieve candidates** (`ber/blocking.py`, `ber/regions.py`, `gpu/dense_retrieval.py`). The country
   is a hard block (it always agrees). The union of these passes:
   - TF-IDF name (character 3-grams), address (words) and combined searches with `sparse_dot_topn`
     top-k;
   - an exact core-name pass;
   - a name search restricted to the same state/region (beats "crowding" by same-named businesses);
   - the **15 nearest neighbours of a fine-tuned dense bi-encoder**.

   Validation pair recall: 0.9955. Every fast top-k is checked against an independent brute force on
   sample rows.
3. **Prune** (`ber/prune.py`): a small LightGBM pre-ranker on cheap signals picks the smallest keep
   rule (top-N + probability floor) that loses ≤ 0.2% of retrieved true pairs. Test candidates go
   from 148.7M to 15.7M.
4. **Features** (`ber/features.py`, `ber/features_extra.py`):
   - RapidFuzz name/address similarities, number agreement and conflicts, retrieval scores and ranks;
   - context within each business's list, name frequency, region match, dense rank/cosine;
   - the probabilities of **two fine-tuned cross-encoders** (e5-small, mDeBERTa), trained only on
     businesses the LightGBM never sees (stacking without leakage).
5. **Match** (`ber/model.py`, `ber/decide.py`): LightGBM with out-of-fold validation. Then **exclusive
   assignment** (each S2/S3 record goes only to the business that scores it highest) and
   rank-dependent thresholds tuned on the out-of-fold macro F0.5 (t1 = 0.62 for the first match,
   t2 = 0.78 for further matches).
6. **Write** (`ber/writer.py`): LF-only TSVs with every format rule self-checked.
   `candidate_pairs.tsv` keeps pairs with matcher p ≥ 0.0001.

## 4. How the submitted file was actually produced (honest lineage)

The submitted files came from our **run 6**, built up over runs 2 to 6. `run_all.py` performs the same
procedure in one pass.

| Part of the submitted run | How it was produced |
|---|---|
| Normalisation, translit dictionary, base training businesses, cross-encoder businesses, final 304,555 businesses | Exactly as `run_all.py` does. The final set was re-created with `ber/queries.py` and is **identical** to the one used. |
| e5 cross-encoder | `ber.cross_encoder train` with the settings above (RTX 3050 laptop, bf16). Its scores were computed in parts across runs 4–6: laptop bf16 plus Kaggle fp16; the two agree to 0.0001 on average. |
| Dense bi-encoder + neighbours | `gpu/dense_retrieval.py` on Kaggle T4 ×2, run by a teammate |
| mDeBERTa cross-encoder | `gpu/mdeberta_cross_encoder.py` on Kaggle T4 ×2, run by a teammate. **Difference:** in our run it scored the candidates of an earlier run (run 4), so run-6 candidates outside that set have no mDeBERTa score (a NaN feature, handled natively by LightGBM). `run_all.py` scores the final candidates instead. |
| Retrieval, pruning, features, LightGBM, decisions, writing | Exactly the code in `src/ber/` (byte-identical to the code of run 6) |

GPU training is not bit-for-bit deterministic, and the mDeBERTa inputs differ as described above. A
re-run therefore reproduces the approach and the score level (validation ≈ 0.988), but not a
byte-identical file. The trained model weights and intermediate score files of our run (about 3 GB)
are available on request.

## 5. Folder layout

```
src/
  run_all.py                 one-command reproduction (steps above)
  ber/                       the pipeline package
    pipeline.py              stages: prepare, regions, pairs, prune, augment, augment_ce, train, predict
    normalize.py translit.py regions.py blocking.py prune.py features.py features_extra.py
    cross_encoder.py ce_data.py ce_export.py queries.py model.py decide.py scoring.py writer.py io_utils.py
  gpu/
    dense_retrieval.py       fine-tuned e5 bi-encoder + top-15 neighbours (Kaggle or local GPU)
    mdeberta_cross_encoder.py  fine-tuned mDeBERTa cross-encoder (train + score, or score only)
  tests/
    test_units.py            pytest unit tests
    make_dev_subset.py       small self-consistent dataset for quick end-to-end checks
```
