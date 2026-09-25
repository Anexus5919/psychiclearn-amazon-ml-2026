# Business Entity Resolution: team PsychicLearn

This pipeline takes the provided train/test files and produces `matching_results.tsv` and `candidate_pairs.tsv`.

- It uses **no external data, APIs or pretrained models**. Every rule and model is learned from, or applied to, the provided files only.
- It runs on CPU only. Tested on Windows 11 and Ubuntu 24.04 with Python 3.12.

## 1. Setup

```bash
cd code/business_entity_resolution
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Reproduce both output files (one command)

The dataset folder must contain `train/` and `test/` exactly as distributed. Plain `.tsv` or gzipped `.tsv.gz` both work.

```bash
cd src
python -m ber.pipeline --data-dir <path>/dataset --work-dir <scratch dir> --out-dir <path>/output
```

The command writes `<out-dir>/matching_results.tsv` and `<out-dir>/candidate_pairs.tsv` with LF line endings. It then self-checks every format rule; the official `utils/validate_submission.py` was also run on the outputs.

**Stages.** Every stage writes to `--work-dir`, so a stopped run resumes where it left off. Use `--stage` to run one stage:

| `--stage` | What it does | Main artefacts in work dir |
|---|---|---|
| `prepare` | Parse the TSVs; normalise names and addresses | `norm/*.parquet` |
| `pairs` | Blocking (candidate generation) and pair features | `pairs/{train,test}/<country>_s<2\|3>.parquet` |
| `train` | Grouped out-of-fold LightGBM, threshold tuning, validation report | `models/fold*.txt`, `report_train.json` |
| `predict` | Score test candidates, apply the decision rule, write both TSVs | `report_predict.json` |

**Main options** (defaults in `ber/pipeline.py:Config`):

| Option | Default | Meaning |
|---|---|---|
| `--n-jobs` | 8 | Worker threads/processes |
| `--train-frac` | 0.08 | Share of train S1 entities used to build training pairs |
| `--k-name`, `--k-addr`, `--k-combo`, `--k-reverse` | 10, 10, 15, 3 | Per-pass candidate budgets |
| `--max-df` | 0.05 | Char-3-grams in more than this share of records are ignored |
| `--n-folds`, `--seed` | 4, 42 | Cross-validation and sampling |

**Hardware used:** a 12-thread laptop with 16 GB RAM. Every stage streams its data in chunks, so peak memory stays at a few GB.

## 3. Method in one screen

1. **Normalisation** (`ber/normalize.py`). Handles:
   - Latin-only accent folding; Indic vowel signs are kept;
   - DBA extraction ("X trading as Y" → Y);
   - website and handle unpacking;
   - honorific and junk removal;
   - canonical legal forms, including dotted French forms such as S.A.R.L.;
   - street-type abbreviations (US / India / France);
   - house-number parsing (zero padding, ordinals, letter suffixes);
   - removal of PO Box / PMB numbers and `null` / `N/A` tokens.
2. **Blocking** (`ber/blocking.py`):
   - Country is a hard block; it agrees on 100% of matched training pairs.
   - Within each country and target source there are three char-3-gram TF-IDF top-k passes: name, address, and name+address.
   - A reverse pass adds each candidate record's top Source-1 records.
   - Every retrieval is asserted against an independent brute-force computation on sampled rows.
   - The union of all passes is exactly the set written to `candidate_pairs.tsv` and scored by the model.
3. **Features** (`ber/features.py`), about 50 in total:
   - name and address similarities (RapidFuzz);
   - house-number agreement and conflict;
   - name and address frequency (generic-name discount);
   - blocking scores and ranks;
   - competition context, i.e. whether another Source-1 record claims this candidate more strongly.
   - Country is **not** a feature, so the model transfers to the unseen country (France).
4. **Model** (`ber/model.py`): LightGBM (MIT licence), grouped 4-fold out-of-fold training by Source-1 entity.
5. **Decisions** (`ber/decide.py`):
   - **Exclusive assignment:** each Source-2/3 record goes to at most one Source-1 entity, a structural fact of the training ground truth.
   - **Rank-dependent thresholds** `t1` (first match) and `t2` (further matches), tuned on out-of-fold predictions to maximise the exact macro F0.5 (`ber/scoring.py`).

## 4. Tests

```bash
pytest tests/test_units.py       # metric (PS worked example), normalisers, decision rule, writer
python tests/make_dev_subset.py --data-dir <dataset> --out-dir <tiny copy>   # smoke-test data
```

## 5. Optional: running on AWS

`src/aws/ec2_job.py` can run the same pipeline as a self-terminating EC2 batch job, with data in S3. It was not used for the final outputs: AWS Free-plan accounts are limited to 2-vCPU / 8 GB instances.
