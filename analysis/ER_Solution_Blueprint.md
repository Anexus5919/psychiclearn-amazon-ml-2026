# Amazon ML Challenge 2026 — Business Entity Resolution
## Evidence-Based Problem Analysis & Maximum-Performance Solution Blueprint

> **Intended reader:** the downstream AI (and the team) that will turn this into the final
> `Documentation_template.md`, `README.md` and methodology write-ups. Completeness and
> traceability are prioritised over brevity, as the master prompt requires.

---

## PART 0 — READ ME FIRST

### 0.1 Source-labelling convention

All tags from the master prompt are used unchanged:

| Tag | Meaning |
|---|---|
| `[PS]` | Official Problem Statement (PDF and the identical `student_resource/README.md`) |
| `[GUIDE]` | Official "Guidelines and Key Instructions" PDF |
| `[AWS-FAQ]` | "AWS Builder Center & AWS Free Tier Instructions" PDF |
| `[UNSTOP]` | Unstop listing. **Not supplied to me**: these statements come only from the master prompt and could not be verified. |
| `[WORKSHOP]` | AWS workshop/blog. **Not supplied to me**: these statements come only from the master prompt and could not be verified. |
| `[BEST-PRACTICE]` | General ML/AWS engineering practice |
| `[STRATEGY]` | Our proposed approach |
| `[INFERENCE]` | A reasoned inference that no source states |
| `[UNSPECIFIED]` | The material does not answer this |

**One new tag is added:**

| Tag | Meaning |
|---|---|
| `[DATA]` | **Measured directly from the provided dataset files** by the scripts in `amazon_ml_2026_analysis/eda/`. Every `[DATA]` number is reproducible (Appendix A). |
| `[LB]` | Observed on the live Unstop public leaderboard (screenshot supplied by the team on 2026-09-25, with about 2 days 10 hours remaining) |

### 0.2 What changed relative to the master prompt

The master prompt (§5) assumed that *no dataset files were supplied* and asked for an EDA **plan**
only. **That assumption no longer holds.** The full dataset is present in
`student_resource/dataset/`, so Part 6 reports **actual measured statistics** (tagged `[DATA]`).
The plan-only checklist is kept for the items not yet measured (§6.11).

### 0.3 Material actually read

| Item | Status |
|---|---|
| `amazon_ml_challenge_2026_master_prompt.md` | Read in full |
| Problem Statement PDF (8 pages; page 8 is blank) | Read in full; identical in substance to `student_resource/README.md` |
| Guidelines PDF (2 pages) | Read in full |
| AWS Builder Center / Free Tier PDF (8 pages) | Read in full |
| `student_resource/README.md` | Read in full |
| `student_resource/Documentation_template.md` | Read in full (§1.9 lists its sections) |
| `student_resource/utils/validate_submission.py` | Read in full, line by line (§1.8 and §1.17) |
| `dataset/train/train_source{1,2,3}.tsv`, `train_ground_truth.tsv` | **Fully scanned** (every row) |
| `dataset/test/test_source{1,2,3}.tsv` | **Fully scanned** (every row) |
| `__MACOSX/`, `.DS_Store` | macOS zip artefacts (resource-fork stubs, 211–354 bytes each), not data. **Exclude them from the submission zip.** |
| Unstop listing, workshop blog/video, PS reference video | **Not supplied**. Content attributed to them is `[UNSTOP]`/`[WORKSHOP]` as quoted by the master prompt, or `[UNSPECIFIED]`. |
| "Original Point 10 A–T structure" (master prompt §11.7) | **Not included in the master prompt** → `[UNSPECIFIED]`. Part 7 follows the master prompt's §10 checklist instead. |

### 0.4 The findings that most change the design (full evidence in Part 6)

1. **Scale is large** `[DATA]`: train has 2.21M S1, 5.03M S2 and 5.29M S3 records. Test has
   **1.73M S1**, 4.89M S2 and 5.08M S3. The raw files total about 2.5 GB. Blocking efficiency and
   memory are first-class engineering problems.
2. **Almost every S1 entity has matches** `[DATA]`: only **5.58%** of train S1 entities are
   singletons. The mean match-list length among non-singletons is **3.67**, and the maximum is 11.
   "Predict all empty" scores just **F0.5 = 0.0558**. Precision-heavy does **not** mean
   "predict little": recall across 3–4 matches per entity is where most of the score lives.
3. **Each S2/S3 record belongs to at most one S1 entity** `[DATA]`. All 7,638,365 matched IDs are
   unique across the ground truth. This **exclusive-assignment constraint** is a large, free
   precision lever.
4. **Country is a perfect hard block in train** `[DATA]`: 100.000% country agreement on all 7.64M
   matched pairs.
5. **Large train→test distribution shift** `[DATA]`: train S1 is 60% US / 40% India. Test S1 is
   **46.8% India / 38.3% US / 15.0% France**, which means 259,452 zero-shot French entities.
6. **Test likely contains more distractors** `[DATA]` + `[INFERENCE]`: S2 and S3 records per S1
   are about 2.28/2.40 in train but about 2.82/2.93 in test, in every country. If test fan-out
   equals train fan-out, the unmatched ("distractor") share of test S2/S3 rises from about 26% to
   about 41%, so there are more chances for false merges.
7. **Postal codes are nearly absent** `[DATA]`: a 6-digit PIN appears in 0.08% of train S1
   addresses and a 5-digit code in 6.7%. **PIN/ZIP blocking, as suggested in the template, cannot
   carry recall.**
8. **Address is at least as discriminative as name** `[DATA]`. On matched pairs, address
   token-set-ratio has a 10th percentile of 70–86, against a 90th percentile of 43–46 on random
   same-country pairs. **8.0% of all matched pairs** have Latin script on both sides yet share
   **no** core name token (invented names, website handles, acronyms). Those are recoverable only
   through the address.
9. **Script mixing is heavy in India** `[DATA]`: about 24% of matched India-S2 names are in a
   non-Latin Indic script (13.9% Devanagari, 9.9% other), as are about 13% of India-S3 names. The
   training data contains the aligned pairs needed to **learn transliteration without any
   external lookup**.
10. **Generic names repeat across different real entities** `[DATA]`: 178K lowercase names occur
    more than once in train S1 ("primary care group" ×253). Test S1 has "bordeaux club sarl" ×205.
    Name-only matching produces false merges.
11. **No leakage via IDs or row order** `[DATA]`: the Spearman correlation between S1 and matched
    S2/S3 numeric IDs or file positions is at most |0.001|. No IDs overlap between train and test.
12. **Measured blocking ceiling** `[DATA]`: crude char-3-gram TF-IDF, top-20 by name ∪ top-20 by
    address, against the full train pools.

    | | Pair recall | Candidates per S1 | Oracle F0.5 | Naive threshold F0.5 |
    |---|---|---|---|---|
    | US | 97.5% | about 77 | **0.992** | 0.748 |
    | India | 92.7% | about 78 | **0.970** | 0.684 |

    - Name-only retrieval saturates around 71% (US) and 57% (India) because generic names crowd
      the top-K.
    - The live public leaderboard top is **0.984** `[LB]`.
    - **Two gaps must be closed:** the matcher and decision layer (0.71 → 0.98) and India
      blocking (lifting the 0.970 ceiling through transliteration and extra passes).
13. **File-format traps** `[DATA]`: all files use **CRLF** line endings, and the test files
    contain RFC-4180-escaped quotes (for example `"""ehpad Club SAS"`). Default pandas parsing is
    correct, but **writing output with Windows default line endings (`\r\n`) silently corrupts the
    last ID of every row** (§1.17).

---

## PART 1 — THE PROBLEM STATEMENT AND OFFICIAL RULES, EXPLAINED IN FULL

Each element below states **what it means**, **why it matters**, and **what goes wrong if it is
ignored**.

### 1.1 Core problem `[PS]`

- **What:** Entity Resolution (ER) across three independent sources of business records that
  share no common identifier and carry noisy name and address fields. For **each Source-1
  entity**, output **all** Source-2 and Source-3 records describing the same real business.
- **Source 1 is the deduplicated reference** `[PS]`: no two S1 rows describe the same business.
  The task is therefore **not** symmetric clustering of all records. It is *S1-anchored,
  one-to-many linking*. Measured consequence `[DATA]`: S2 and S3 are **not** deduplicated. One
  business commonly has 2–4 records inside the same source (the (S2=2, S3=2) pattern alone covers
  9.4% of S1 entities).
- **Zero, one or many** `[PS]`: an S1 entity can have no match (a *singleton*). Measured `[DATA]`:
  0 matches 5.58%, 1 match 5.40%, 2 matches 17.00%, 3 matches 24.05%, 4 matches 21.94%,
  5 matches 14.59%, 6 or more 11.43% (maximum 11).
- **If ignored:** a pipeline that predicts one best match per S1 caps recall at about
  1/3.67 ≈ 27% for the typical entity, which is disastrous even under F0.5.
- **Reference video:** its contents are `[UNSPECIFIED]` (not transcribed or supplied).

### 1.2 File format `[PS]`

- **What:** every input and output file is **tab-separated** `.tsv`, because addresses and ID
  lists contain commas. Read with `sep="\t"` explicitly.
- **Measured detail `[DATA]`:**
  - Every line in every file ends in **CRLF** (`\r\n`).
  - Test files contain fields with CSV-escaped double quotes: 134 lines in test S1, 349 in test S2
    and 330 in test S3 (plus 4 and 6 in train S1 and S2). An example is
    `"""ehpad Club SAS"` → `"ehpad Club SAS`.
  - Pandas' default quoting (`QUOTE_MINIMAL`) decodes these correctly. Row counts match
    `wc -l − 1` exactly under both default and `QUOTE_NONE` parsing.
  - All rows have exactly 4 fields (2 in the ground truth). There is no BOM.
- **If ignored:** omitting `sep="\t"` yields one column. Writing a *comma*-separated output fails
  validation. Writing CRLF output is the subtle trap described in §1.17.

### 1.3 Data schema `[PS]`

| Column | Meaning | Measured reality `[DATA]` |
|---|---|---|
| `entity_id` | Unique ID; the prefix `S1-`/`S2-`/`S3-` encodes the source | Unique within every file. Numeric part is 1–9 digits (total length 4–12). **Random**: no correlation with matches. |
| `business_name` | Abbreviations, legal suffixes, typos, transliterations | Never empty. S2/S3 add all-caps, native scripts, handles, DBA forms and more (§6.6). |
| `business_address` | Partial, reordered, abbreviated, landmark-based | Never empty in S1. **Empty in 3.3% of train S2/S3** and 2.7% of test S2/S3. Literal `null`/`N/A`/`NULL` tokens appear in about 1.7%. |
| `country` | Open set of string labels | Train: {US, India}. Test: {India, US, **France**}. **Always identical within a matched pair** (train). |

- **France is unseen in training** `[PS]`. Treat country as an **open-set string**: never
  hard-code, filter, or one-hot encode to {US, India}. Every French test entity must still get a
  row (possibly empty).
- **If ignored:** a pipeline that one-hot encodes country with a fixed vocabulary crashes on the
  test set or silently drops 15% of the entities, and missing entities mean **rejection**.
- **No source column** `[PS]`: derive the source from the ID prefix.

### 1.4 Ground truth (training only) `[PS]`

`train_ground_truth.tsv` has `source1_entity_id` and `matched_entity_ids` (comma-separated,
empty for singletons). Measured `[DATA]`:

- It covers **all** 2,206,821 train S1 IDs exactly once. Row order is unrelated to S1 file order
  (Spearman 0.0003).
- 7,638,365 matched IDs: S2 3,693,619 and S3 3,944,746. **All unique**, so the S1→{S2,S3} relation
  is a *partial function from S2∪S3 to S1*.
- 73.4% of S2 records and 74.6% of S3 records are matched. The rest are **distractors** with no S1
  entity.

### 1.5 Noise patterns called out by the PS `[PS]`, and confirmed `[DATA]`

The PS lists these noise types:

- **Name:** abbreviations (Corp/Corporation, Pvt/Private, Ltd/Limited), legal-suffix
  inconsistency, DBA/trade names, `&` vs "and", word-order transposition, typos.
- **Address:** abbreviations (Rd/Road, St/Street), transliteration, missing components (no PIN or
  state), landmark references ("Near SBI ATM"), municipal numbering formats, component reordering.

**All of these are confirmed in the data.** Several additional noise types exist that the PS does
not mention: invented replacement names, website/handle forms, acronym-only names, honorific
prefixes, junk punctuation prefixes, random accent injection, city-name substitution, `Street`
mis-expanded to `Saint`, house-number digit drops and zero-padding, injected PO Box/PMB numbers,
and native-script state names. The full taxonomy with real examples is in §6.6.

### 1.6 Dataset composition `[PS]`

- Train: three source files plus the ground truth. Test: three source files, **no labels**.
- Teams must build their own validation split and score it with the exact F0.5 definition
  (§1.11). §4.7 designs this carefully because of the train→test shift (§0.4 items 5–6).

### 1.7 Required outputs `[PS]`

**(a) `output/matching_results.tsv`, the only leaderboard-scored file.**

- Header exactly `source1_entity_id<TAB>matched_entity_ids`.
- **Exactly one row per test S1 entity**: 1,732,544 rows `[DATA]`.
- Empty `matched_entity_ids` for predicted singletons.
- No duplicate IDs within a list. Only S2-/S3- IDs **that exist in the test files**.
- IDs are comma-separated with no quoting and no spaces (implied by the example).

**(b) `output/candidate_pairs.tsv`, not scored, but audited.**

- Header `source1_entity_id<TAB>candidate_entity_ids`. Same row and format rules.
- It must be the **final** candidate set that the matching model actually scores at inference:
  the *last* blocking or filtering stage, not an early pass.
- **Every matched ID must also be a candidate** (matches ⊆ candidates). The organisers use this
  file to measure blocking *recall ceiling* and *reduction ratio* and to verify the pipeline
  `[PS]`.
- **If ignored:**
  - Reporting an early, wider candidate set misrepresents the pipeline, which is a fair-play and
    reproducibility risk at review.
  - Any match outside the candidate set flags a pipeline bug.
- **Size planning `[INFERENCE]`:** about 30 candidates × 1.73M S1 × about 13 bytes ≈ 0.7 GB
  uncompressed. It compresses well in the zip.
- **Design implication `[STRATEGY]`:** if a cheap *pre-ranker* prunes the blocking union before
  the main model, the file must contain the **pruned** set. The documentation must describe the
  pre-ranker honestly as part of candidate generation (§4.3).

### 1.8 Validation tooling `[PS]`

`utils/validate_submission.py` is stdlib-only. It checks:

- header correctness;
- detection of comma-separated files;
- duplicate S1 rows;
- intra-list duplicates;
- S1 self-matches;
- wrong prefixes;
- missing required S1 rows;
- extra S1 rows not in the test set;
- *optionally* (`--check-ids`) that every ID exists in test S2/S3.

It **warns** (never fails) when matches ⊄ candidates. It prints `PASS` with exit code 0 or a
numbered list with exit code 1. It **does not compute the score**. Run it from `student_resource/`:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Behaviours found by reading the code (these matter; see §1.17):

- It lowercases and strips header cells before comparing, so its header check is lenient.
- The ID-existence check is **off by default** because it loads about 10M IDs into memory
  (a few GB).
- It parses each row with `rest.rstrip("\n").split(",")`, so a trailing `\r` stays attached to
  the last ID and **is not detected** unless `--check-ids` is on.

### 1.9 Final submission package `[PS]`

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/                 # all source
│       ├── README.md            # exact end-to-end reproduction: data → blocking → matching → output
│       └── requirements.txt     # pinned versions
└── Documentation_template.md    # filled-in methodology (.md, or .pdf export)
```

- `code/` must regenerate **both** output files from the train and test data *using only what is
  in that folder* `[PS]`. Trained artefacts must either be regenerated by the code or shipped in
  the zip. The PS does not say which is preferred (`[UNSPECIFIED]`). Recommended `[STRATEGY]`:
  - the code must be able to retrain end-to-end with a fixed seed;
  - optionally ship the final model files (small for GBDTs) so that inference-only reproduction is
    fast;
  - pretrained weights are downloaded by pinned name and revision.
- The template sections are: Executive Summary; Methodology (Problem Analysis, Solution Strategy);
  Candidate Generation (keys, number of pairs, how true matches were protected); Matching Model
  (features, model type, threshold method); Results & Error Analysis (macro F0.5, common false
  positives and negatives); Conclusion; Appendix A (code structure and entry points); Appendix B
  (extra results). Teams may adapt sections `[PS]`.
- Do **not** include `__MACOSX/` or `.DS_Store` `[BEST-PRACTICE]`.

### 1.10 Hard constraints `[PS]`

1. The format must match exactly. Only a **`SCORED`** status with an F0.5 counts. A failed
   validation means the submission is not evaluated.
2. Matched IDs must be **S2/S3 IDs present in the test set**. S1 self-matches and non-existent IDs
   → **rejected**.
3. **Every** test S1 entity must appear. Missing entities → rejection.
4. No duplicate IDs inside a list and no duplicate `source1_entity_id` rows → otherwise rejection.
5. **The final model must be MIT- or Apache-2.0-licensed and have at most 8 billion parameters.**
   - Apply this to **every pretrained component that influences predictions**: embedding models,
     cross-encoders, transliteration models, any LLM. Not only the "main" classifier
     `[STRATEGY]`.
   - The PS wording is "should be". We treat it as **hard** because the organisers review model
     licences in the zip.
   - Whether the 8B limit applies per model or to an ensemble total is `[UNSPECIFIED]`. The
     conservative reading is that the **total** stays ≤ 8B.
   - GBDT models (LightGBM: MIT; XGBoost and CatBoost: Apache-2.0) comply trivially.
   - Library licences (BSD scikit-learn, and so on) are not "models", but keeping everything
     permissive is safest.

### 1.11 The evaluation metric: F0.5, understood deeply `[PS]`

**Definition** `[PS]`:

`F0.5 = 1.25·P·R / (0.25·P + R)`, computed **per S1 entity** and **macro-averaged over all S1
entities** in the evaluation subset. Singletons are included:

- a correct empty prediction scores **1.0**;
- any non-empty prediction on a singleton scores **0.0**.

**Equivalent count form** (derived; algebraically identical): with TP, FP and FN counted per entity,

`F0.5 = 1.25·TP / (1.25·TP + 0.25·FN + FP)`

**One false positive costs as much as four false negatives** in the denominator. This form makes
every design decision below computable.

**Worked example reproduced** `[PS]`: predicted {S2-00047, S2-00193, S3-00812}, true {S2-00047,
S3-00812}. TP=2, FP=1, FN=0, so F = 2.5/(2.5+0+1) = **0.7143**. P = 2/3, R = 1, and
1.25·0.667/(0.1667+1) = 0.714 ✓.

**Edge cases:**

| Situation | Score | Source |
|---|---|---|
| True set non-empty, prediction empty | P = 0/0 (undefined), R = 0 → **0.0** | `[INFERENCE]`, standard convention; the count form gives 0 |
| True set non-empty, prediction non-empty with TP = 0 | **0.0** | `[PS]` formula |
| True set empty, prediction empty | **1.0** | `[PS]` |
| True set empty, prediction non-empty | **0.0** | `[PS]` |

**What macro-averaging implies:**

- Every S1 entity weighs the same. An entity with 11 true matches counts no more than one with 1.
  Scoring is **entity-centric**, not pair-centric: 1,000 correct pairs concentrated in a few
  entities are worth less than getting many entities "mostly right".
- **Singleton economics** `[DATA]` + derived:
  - Singletons are 5.58% of train entities, and the test proportion is `[UNSPECIFIED]`.
  - A false merge on a singleton costs the full 1.0 for that entity.
  - A missed first match on a non-singleton also costs 1.0 (score 0).

**Rank-dependent optimal thresholds (derived from the count form; a key design insight):**
Consider an entity with 4 true matches whose candidates are ranked by probability of being
correct, p.

| Current prediction | Add a correct ID | Add a wrong ID | Break-even p |
|---|---|---|---|
| empty (F = 0) | 0.625 | 0 | ≈ P(singleton)/0.625, **≈ 0.09** at the 5.6% prior |
| 1 correct (0.625) | 0.833 | 0.417 | **0.50** |
| 2 correct (0.833) | 0.9375 | 0.625 | **0.67** |
| 3 correct (0.9375) | 1.000 | 0.750 | **0.75** |

- **The first candidate should be emitted at a low probability.** Subsequent candidates need
  progressively higher confidence.
- One global threshold is therefore **suboptimal**. §4.6 turns this into an expected-F0.5
  set-selection rule on calibrated probabilities.
- This is the correct reading of "precision-heavy". It does **not** mean "high uniform threshold".

**β = 0.5 rationale** `[PS]`: merging two distinct businesses is considered worse than missing a
link.

### 1.12 Official "Tips for Success" `[PS]`

These are official statements, not our strategy:

1. Invest in blocking, because it sets the recall ceiling.
2. Use string-similarity features (Jaccard, Levenshtein, TF-IDF cosine) for name and address.
3. Attend to country-specific address patterns.
4. Weigh the precision–recall trade-off (F0.5).
5. Do not neglect singletons.
6. Validate the output format before submitting.

Data-driven refinement `[DATA]` + `[STRATEGY]`:

- (1) must include **address-based and transliteration-aware** blocking, not name-only blocking.
- (3) must extend zero-shot to France.
- (4) needs the rank-dependent view in §1.11.

### 1.13 Academic integrity and fair play `[PS]`, strict

- **Banned:**
  - commercial entity-resolution APIs;
  - government business-registry lookups;
  - geocoding APIs for address normalisation;
  - **any external data augmentation from internet sources**.
- Enforcement: code and methodology review. Any evidence of external lookup means **immediate
  disqualification**. "Designed to test … skills using **only the provided training data**."
- **Architectural consequences `[STRATEGY]`:**
  - Every normaliser, dictionary and mapping must be either (a) hand-written generic linguistic
    rules (abbreviation expansion, Unicode folding) or (b) **learned from the provided train/test
    files**.
  - No calls to maps, geocoders, registries or hosted LLM/NER APIs at any stage.
  - General-purpose pretrained models (MIT/Apache, ≤8B) run **locally** and are allowed as models.
    They are not "lookups".
- **Grey areas, declared openly `[STRATEGY]`:**
  - (i) Fitting unsupervised statistics (IDF, vocabularies, co-occurrence maps) on the **unlabeled
    test inputs**. This is transductive, uses no labels, and is standard ER practice.
  - (ii) Pseudo-labelling test pairs.
  - (iii) A small hand-written table of French legal forms and street-type abbreviations. This is
    generic linguistic knowledge, not business identity data.
  - Avoid any hand-typed *geographic reference table* (for example a French region↔département
    map) and learn such equivalences from the data instead (§4.2).
  - Document every grey-area choice explicitly in the methodology.
- One registration per person; no multiple IDs `[GUIDE]`.

### 1.14 Official competition guidelines `[GUIDE]`

- **Window:** 25 Sep 2026, 00:00 IST → 27 Sep 2026, 23:59 IST. The dataset and PS are released on
  day 1, and submissions close at the end of day 3. Per the master prompt, this window does **not**
  bound the technical design here.
- A live leaderboard runs during the challenge, with a final leaderboard afterwards. Queries go
  through a Google Form (link not reproduced).
- **Artefacts for the team's best submission:**
  - a **1–2 page** document (ML approach, models, experiments, conclusion);
  - source code for experiments, training and inference, with comments describing the functions.
- **At most 5 submissions per day, over 3 days (15 in total).** The button disables after that.
- **Maintain the version history of all submissions**: shortlisting uses the submitted solutions,
  and final source code may be requested later.
- **Two leaderboards, public and private.** Shortlisting considers both.
  - The PS adds that final ranking is **private-only** `[PS]`.
  - Public is a subset of test S1 and private is the remainder. Predictions are always for the
    full test set.
- The **top 100** are announced after artefact checks, leaderboard score and eligibility. They
  must then submit:
  - methodology;
  - candidate generation / blocking strategy;
  - model architecture and feature engineering;
  - other relevant information.

  This overlaps the zip's methodology document. Treating it as a possible second, expanded
  write-up is `[INFERENCE]`.
- Desktop or laptop only. **No simultaneous logins**: one device per participant. Detection may
  terminate the attempt.
- Technical issues: clear the cache, try another browser or incognito, try another network, or
  email support@unstop.com with a screenshot and the registered email. Support will not make
  decisions for participants.
- Cheating, plagiarism, or multiple-ID attempts → instant disqualification.

### 1.15 AWS Builder Center, Student Rewards and Free Tier `[AWS-FAQ]`

**Builder ID and alias**

- The Builder ID is a personal credential, **separate** from AWS accounts or the console.
- Sign-up: email → verification code → name → password → optional captcha.
- First sign-in asks for an **alias**:
  - 3–19 characters;
  - starts with a letter;
  - lowercase a–z and 0–9 only;
  - no spaces or special characters;
  - must follow the Builder Terms.
- Find the alias via name → Manage Profile.
- A valid alias is mandatory for registration `[UNSTOP]`, and an invalid alias leaves registration
  incomplete `[UNSTOP]`.

**Student Rewards**

- Verify through SheerID, then complete the profile (photo and About section). This earns the
  "Photo Finisher" and "Hello, World!" badges and a **12-month Skill Builder subscription**.
- **7 badges → $10 credits; 14 badges → $20 credits; 21 badges → $100 certification voucher**
  (AWS Certified Cloud Practitioner).
- Verification takes minutes, or 24–48 hours if SheerID asks for documents.
- Common rejection reasons: ineligible institution, unclear document, stale enrollment dates.
  Retry with other documents.
- The master prompt's "$579 total / $449 subscription value" figures are `[WORKSHOP]` and not in
  the AWS-FAQ PDF.

**Free Tier**

- New accounts get up to **$200 in credits**: $100 at sign-up plus up to $100 for exploring
  services. The Free plan lasts up to **6 months** and includes 30+ always-free services.
- During the ML Challenge, per the PDF:
  - **SageMaker (2-month trial):** 250 h `ml.t3.medium` notebook and 50 h `ml.m4.xlarge` or
    `ml.m5.xlarge` training.
  - **S3:** 5 GB storage, 20,000 GET, 2,000 PUT per month.
- The master prompt's "125 h `ml.m5.xlarge` inference", "Lambda 1M requests" and "~$0.12/h idle
  endpoint" are `[WORKSHOP]` and **not** in the PDF.
- The top 500 teams on the leaderboard get extra credit codes, redeemed via Billing and Cost
  Management → Credits → Redeem credit. The "48-hour mark" timing is `[UNSTOP]`/`[WORKSHOP]`.

**Cross-account credit pooling (documented method)**

1. Exchange 12-digit account IDs.
2. Member A creates `s3://hackathon-<team-name>` and uploads artefacts.
3. Member A adds a bucket policy granting `arn:aws:iam::<ID>:root` the `s3:GetObject` and
   `s3:ListBucket` permissions on the bucket and `/*`.
4. Member B copies the artefacts into their own bucket and continues.

Alternatives:

- **Option 2:** use the SageMaker training output at
  `s3://sagemaker-<region>-<account-id>/output/<job>/output/model.tar.gz`, then
  `sagemaker.model.Model(model_data=..., role=..., image_uri=<same container>)`.
- **Lazy option:** `aws s3 presign … --expires-in 86400`.

Monitor usage via Billing → Free Tier, and hand off to a teammate before hitting the limits.

**Documented caveats (verbatim substance)**

- Training jobs are **not** shareable across accounts. Only artefacts move, and a job cannot be
  resumed in another account.
- **Endpoints can't be shared cross-account without AWS Organizations.** The PDF says this is "not
  a problem here since you only need to submit a CSV, not host an endpoint".
- A custom KMS key breaks cross-account decryption. Use default S3 encryption or none.
- The Free plan auto-closes after 6 months or when the $200 is exhausted. The SageMaker 2-month
  trial window is shorter.
- Teammates must use the **same framework version and container**.
- Bucket names are global.
- **Cross-region copies cost credits**, so keep every account in one region.

The master prompt's reference-document list matches the PDF's link titles. Their contents are
`[UNSPECIFIED]`.

### 1.16 Unstop and workshop context: stated by the master prompt, unverified here

The following could not be checked against any supplied file. They are reproduced so nothing is
dropped:

- **Team and eligibility** `[UNSTOP]`:
  - teams of 2–4, cross-college allowed;
  - B.E./B.Tech/M.Tech/M.S./PhD students graduating in 2027 or 2028;
  - a Builder Center profile ID is required;
  - **only the team leader can submit**.
- **Timeline** `[UNSTOP]`:
  - registration 7–22 Sep;
  - best-practices session 21 Sep;
  - hackathon 25–27 Sep;
  - top-50 results 2 Oct;
  - Grand Finale 7 Oct, where the top 10 present to Amazon scientists (selection uses the
    leaderboard **and** the Round-1 documentation).
- **Prizes** `[UNSTOP]`:
  - ₹1,00,000, ₹75,000 and ₹50,000;
  - top 50 → Amazon pre-placement interviews;
  - all participants → $200 credits;
  - top 500 → +$100;
  - top 10 and the top 10 women-only teams → certificates and swag.
- **Workshop guidance** `[WORKSHOP]`:
  - Studio needs a Domain; a Notebook Instance does not.
  - Local training suits small data; training jobs suit large data or GPUs.
  - Local prediction versus an endpoint.
  - **"Notebook Instance + local training + local prediction. You submit a CSV, not a running
    API."**
  - The churn-XGBoost demo pattern: setup → EDA → feature engineering → 67/22/11 split → local
    `xgb.train` → predict at 0.5 → metrics → `save_model`.
  - Stop, don't delete, notebooks overnight.
  - Set billing alerts, prefer `us-east-1`, and delete idle endpoints.
  - The "no header row, target column first" convention applies **only** to the SageMaker built-in
    XGBoost container, not to the Python `xgboost` library.
  - The churn demo is a *binary classification toy*. Only its operational pattern transfers to ER;
    its feature engineering does not.

### 1.17 Conflicts, ambiguities and traps found in the official material

| # | Finding | Evidence | Resolution `[STRATEGY]` |
|---|---|---|---|
| C1 | **Doc length conflict**: GUIDE says "1–2-page document"; PS says "no page limit — prioritise clarity and technical depth" | `[GUIDE]` vs `[PS]` | Ship the full `Documentation_template.md` in the zip (PS governs the zip). Also keep a 1–2-page executive version ready for the GUIDE artefact and the top-100 follow-up. |
| C2 | **Non-existent IDs**: PS says they "will be rejected"; the validator docstring says "a missing/garbage matched ID only lowers your score, never rejects your submission" | `[PS]` vs validator | Treat the PS as binding. **Always run the validator with `--check-ids`** before uploading (drop `--candidate` if memory is short). |
| C3 | **CRLF trap**: the validator parses `rest.rstrip("\n").split(",")`. A CRLF-terminated row leaves `\r` on the last ID. The prefix check still passes, so without `--check-ids` the error is invisible. Pandas `to_csv` on Windows writes `os.linesep` = `\r\n` by default. The scorer's parsing is `[UNSPECIFIED]`: at best the last ID of every row is wrong (silent score loss), at worst it is rejected. | Code reading + `[DATA]` (the inputs are CRLF) | Write with `open(path, "w", encoding="utf-8", newline="\n")` or `to_csv(..., lineterminator="\n")`. Assert `b"\r" not in file_bytes` before every upload. |
| C4 | The subset rule (matches ⊆ candidates) is a *warning* in the validator but a stated expectation in the PS | Validator vs `[PS]` | Enforce it in our own writer: intersect before writing and fail the run if violated. |
| C5 | The validator's header check lowercases and strips; the scorer may be strict | Code reading | Emit the exact lowercase header with no extra columns. |
| C6 | "Final model … up to 8B": per model or in total? "should" or "must"? | `[PS]` wording | Conservative: total ≤ 8B, and treat it as mandatory. |
| C7 | Are pretrained weights "external data"? | `[PS]` bans *lookups* and *data augmentation* | Pretrained, general-purpose MIT/Apache models are explicitly anticipated by constraint 5. Use them offline, never as a lookup service. |
| C8 | Public/private split: "a subset" of the test set, with an unknown size and stratification | `[PS]` | Do not tune thresholds on the public LB. Rely on validation (§4.7). |

---

## PART 2 — TEAM-LEVEL OVERALL OUTLINE

### 2.1 The solution in one paragraph `[STRATEGY]`

1. **Partition by country.** This is a hard block, exact in train (§6.4).
2. **Normalise names and addresses aggressively.** This includes transliteration of Indic scripts
   into Latin **learned from the aligned training pairs**, legal-form canonicalisation, handle and
   website unpacking, address-component parsing, and state/region canonicalisation learned from
   the data.
3. **Generate candidates with several complementary, bidirectional retrieval passes:**
   - name char-n-gram TF-IDF;
   - address TF-IDF;
   - multilingual dense embeddings;
   - exact normalised keys;
   - searches both S1→pool and pool→S1.
4. **Prune** that union with a cheap ranker to a fixed budget, and write that set to
   `candidate_pairs.tsv`.
5. **Score each pair with a gradient-boosted classifier.** Its inputs are about 80 similarity,
   parsing, frequency and *context/competition* features, optionally stacked with a fine-tuned
   multilingual cross-encoder.
6. **Calibrate** the scores.
7. **Decide the final sets under the data's structural constraints:**
   - each S2/S3 record is assigned to at most one S1 (exclusive assignment);
   - per-entity sets maximise *expected F0.5* using rank-dependent thresholds;
   - graph consistency uses S2↔S3 sibling links.
8. **Harden for France** with country-agnostic features, leave-one-country-out validation and
   pseudo-labelling.
9. **Measure everything** on a validation design that reproduces the test's country mix and
   distractor density.

### 2.2 Pain points → design responses

| Pain point | Evidence | Response (section) |
|---|---|---|
| 1.73M × 10M comparison space | `[DATA]` | Country partition + multi-pass top-K retrieval + pruning (§4.3) |
| Recall ceiling set by blocking | `[PS]` | Union of name, address, embedding and key passes; bidirectional; measured pair-completeness per pass (§4.3, §6.9) |
| Precision-heavy metric **and** about 3.7 matches per entity | `[PS]` + `[DATA]` | Calibrated probabilities + expected-F0.5 set selection + exclusive assignment (§4.6) |
| Singletons, 5.6% | `[DATA]` | Explicit "empty" option in set selection; singleton-probability features (§4.6) |
| Generic names repeated across entities | `[DATA]` | Name-frequency features; address evidence required when the name is generic (§4.4) |
| Invented, handle, acronym or DBA names | `[DATA]` | Address-driven blocking and features; handle/website splitter; DBA extraction; acronym features (§4.2, §4.4) |
| Indic scripts (about 24% of India-S2 matches) | `[DATA]` | Transliteration dictionary + character model learned from train pairs; multilingual embeddings (§4.2) |
| France zero-shot (15% of test) | `[PS]` + `[DATA]` | Country-agnostic features; transductive IDF; LOCO validation; pseudo-labels (§4.8) |
| More test distractors (inferred) | `[DATA]`/`[INFERENCE]` | Distractor-density-matched validation; competition features; exclusive assignment (§4.7) |
| No external lookups allowed | `[PS]` | Everything learned from the provided files; offline models only (§1.13) |
| Memory limits (16 GB laptop; free-tier notebook has 4 GiB) | Environment + `[AWS-FAQ]` | Per-country/per-source streaming; pyarrow; a larger instance for heavy passes (§5.3) |
| Only 15 submissions; public LB is a subset | `[GUIDE]`/`[PS]` | Validation-driven decisions; submission ledger (§3.3) |
| CRLF / format traps | `[DATA]` | LF writer + byte-level assertions + validator with `--check-ids` (§4.8) |

### 2.3 Roles for a team of 2–4 `[STRATEGY]`

| Role | Owns | Key deliverables |
|---|---|---|
| **A — Data & normalisation** | L1, L2 | Loader, normalisers, learned transliteration and state dictionaries, unit tests on real noisy samples |
| **B — Retrieval & infrastructure** | L3, AWS | Blocking passes, pruning ranker, recall/RR dashboards, S3 layout, SageMaker jobs, cost guardrails |
| **C — Modelling & decisions** | L4, L5, L6 | Feature factory, GBDT/cross-encoder, calibration, assignment, set selection |
| **D — Evaluation, ops & docs** (the leader, since only the leader can submit `[UNSTOP]`) | L7, L8, L9 | Exact scorer, fold design, submission ledger, validator runs, packaging, documentation |

- With 2 people: merge A+B and C+D.
- Interfaces between roles are **files with fixed schemas** (Parquet). Every stage is rerunnable
  on its own.

---

## PART 3 — PHASE-BY-PHASE PLAN (kickoff → final submission)

No step is dropped for time reasons. §3.2 tiers the steps afterwards, as the master prompt allows.

### Phase 0 — Environment, compliance and bookkeeping

1. Create the repository `code/business_entity_resolution/` with `src/`, `configs/`, `tests/`,
   `README.md` and `requirements.txt`, under git. Tag every submission (`sub-YYYYMMDD-N`).
2. Set up the Python 3.11/3.12 environment. On the local machine, **install a CUDA build of
   PyTorch**: the currently installed `torch 2.11.0+cpu` cannot use the RTX 3050 `[DATA]`
   (environment check).
3. **Licence register** (`LICENSES.md`): every pretrained model with its licence, parameter count,
   pinned revision and source URL. Every row must be MIT or Apache-2.0 and within budget.
4. **Submission ledger** (`submissions.csv`): timestamp, git tag, config hash, validation F0.5
   (per country and test-mix weighted), public LB score, notes. This satisfies the GUIDE's
   version-history requirement.
5. AWS:
   - one account per member, same region;
   - billing alarms;
   - an S3 bucket layout (§5.3);
   - dataset upload;
   - least-privilege IAM role.
6. Freeze the fair-play rules into a checklist, and **grep the code for network calls** before
   packaging.

### Phase 1 — Data audit (largely **done**, Part 6)

1. Parsing and format facts.
2. Profiles.
3. Ground-truth structure.
4. Leakage audit.
5. Pair signals.
6. Baseline blocking recall.

Remaining items are in §6.11, most importantly:

- address sharing across distinct S1 entities (hard negatives);
- S2↔S3 sibling similarity;
- the Indic token out-of-vocabulary (OOV) rate of the test set against the learned dictionary;
- France-specific profile.

### Phase 2 — Evaluation harness (before any model work)

1. Implement the **exact scorer** (§4.7), with unit tests covering the PS worked example (0.714),
   singleton cases and the empty-prediction case.
2. **Fold design:**
   - 5-fold split over S1 entities, stratified by country × match-count bucket;
   - each fold's pool is its matched S2/S3 records plus a proportional random share of distractors;
   - a **"dense-distractor" variant** raises the distractor share to the test-implied level
     (§4.7).
3. **Test-mix weighting:** report the per-country score and a weighted estimate using the test S1
   mix (India 0.4675, US 0.3827, France 0.1497, with France proxied by LOCO).
4. **LOCO protocol:** train on US and evaluate on India, and train on India and evaluate on US.
   This is the France proxy.
5. Report blocking metrics (pair completeness, reduction ratio, candidates per S1) and the
   entity-level recall ceiling (the oracle F0.5 given the candidates).

### Phase 3 — Baselines and the first submission

1. **B0:** all empty (0.0558 on train).
2. **B1:** exact normalised name + country.
3. **B2:** TF-IDF name+address threshold (the §6.9 naive baseline).
4. Produce B2 on test **→ submission #1.** Its purposes:
   - verify format, SCORED status and the LF writer;
   - learn how validation and the public LB relate;
   - validate the whole I/O path end-to-end.

### Phase 4 — Normalisation library (L2)

1. Name normaliser:
   - Unicode NFKC;
   - case-folding;
   - accent folding for Latin script only;
   - junk-prefix stripping;
   - bracket handling;
   - honorific removal;
   - legal-form extraction into a separate field;
   - DBA/aka/formerly split;
   - handle and website unpacking (a word-segmentation dictionary built from the corpus);
   - acronym generation.
2. Transliteration:
   - learn a **token dictionary** by aligning native-script and Latin names in train pairs;
   - learn a **character model** for OOV tokens, trained on the same pairs;
   - add a rule-based fallback.
3. Address normaliser:
   - null-token removal;
   - PO Box/PMB removal;
   - street-type canonicalisation (US, India and France abbreviations);
   - ordinal words ↔ digits;
   - house-number parsing (range, suffix and zero-padding handling);
   - landmark phrase tagging;
   - **state/region canonicalisation learned from co-occurrence** (S1 `TX` ↔ S3 `Texas`; native
     `महाराष्ट्र` ↔ `Maharashtra`);
   - city extraction.
4. Unit tests built from the real noisy examples in §6.6.

### Phase 5 — Blocking (L3)

1. Implement each pass independently and measure its pair completeness per country and source:
   - (a) name char-3/4-gram TF-IDF top-K;
   - (b) address TF-IDF top-K;
   - (c) the combined name+city string;
   - (d) exact keys (normalised name, house number + street token, sorted-token keys);
   - (e) multilingual dense embedding approximate nearest neighbours (ANN);
   - (f) **the reverse direction** (pool→S1 top-k).
2. Union the passes. Measure the **marginal recall gain** of each pass to decide which are worth
   their cost.
3. Train the **pruning pre-ranker** (a small GBDT on cheap features). Pick the per-S1 budget N
   where the oracle F0.5 flattens.
4. Freeze the candidate generator and write `candidate_pairs.tsv` for validation and test.

### Phase 6 — Features and matcher (L4, L5)

1. Feature factory (vectorised, multiprocess), about 80 features (§4.4).
2. LightGBM with out-of-fold training. Candidates for hard negatives are the non-matching
   blocking candidates, which is the realistic distribution.
3. Calibration (isotonic or Platt on OOF). Report the reliability curve.
4. Error analysis → iterate on features and normalisers.

### Phase 7 — Decision layer (L6)

1. Exclusive assignment of each S2/S3 record to at most one S1.
2. Rank-dependent thresholds, then full expected-F0.5 set selection.
3. Graph consistency using S2↔S3 sibling edges.
4. Tune everything on OOF, fold by fold.

### Phase 8 — France hardening

1. LOCO experiments to choose country-agnostic features and to test transductive IDF.
2. A French rule pack: legal forms, street types, `N°`, bis/ter, and data-learned
   region↔département equivalence.
3. Pseudo-labelling rounds on France test pairs, validated by simulating the procedure on LOCO.

### Phase 9 — Neural components

1. Choose a multilingual embedding model (MIT/Apache).
2. Encode all names and addresses; use them for ANN blocking and cosine features.
3. Fine-tune a cross-encoder on hard pairs and apply it to the **uncertain band** only.
4. Stack it into the GBDT as a feature, trained on OOF.
5. Ablate: keep it only if the test-mix-weighted validation improves.

### Phase 10 — Test inference and submission loop

1. Run the full pipeline on test and write both files.
2. Byte-level checks, then the validator with `--check-ids`, then an upload.
3. Record the result in the ledger.
4. Compare the public LB with the validation estimate. Investigate if the gap exceeds the expected
   noise.

### Phase 11 — Robustness and ablation

- Seed variance.
- Ablations of each pass, feature group and decision component.
- Sensitivity to the distractor-density assumption.
- The final choice favours **robust** configurations over the public-LB maximum.

### Phase 12 — Packaging and documentation

1. A clean-environment reproduction dry run (a fresh venv or SageMaker instance, `pip install -r
   requirements.txt`, one command → both outputs). Compare file hashes.
2. Fill in `Documentation_template.md` from the ledger and the ablation tables.
3. Prepare the 1–2-page summary.
4. Build the zip without `__MACOSX` and `.DS_Store`, then validate the zip contents.

### Phase 13 — Post-round (top 100)

- Expand the methodology, blocking, model and feature sections from the same material.

### 3.2 Tiering (after the ideal design, as allowed)

| Tier | Components |
|---|---|
| **Must-have core** | Exact scorer + folds; country partition; normalisers (without the learned transliteration); name TF-IDF + address TF-IDF bidirectional blocking + pre-ranker; LightGBM with string, address and context features; calibration; exclusive assignment; rank-dependent thresholds; LF writer + validator |
| **Strongly recommended** | Learned transliteration dictionary + character model; learned state canonicalisation; handle/website segmentation; expected-F0.5 set selection; dense-distractor validation; LOCO validation; French rule pack; embedding-based blocking pass |
| **Optional / experimental** | Cross-encoder on the uncertain band + stacking; S2↔S3 graph consistency; France pseudo-labelling rounds; auxiliary sibling model; LLM adjudication of a few thousand borderline pairs (≤8B, Apache/MIT, local; low expected value per unit cost) |

### 3.3 Submission-budget policy (15 total) `[STRATEGY]` on `[GUIDE]` facts

- **Day 1:** 1–2 submissions (the baseline plus the first real model) to verify format and measure
  how the LB relates to validation.
- **Day 2:** 2–4 submissions, each a *validated* step change (blocking v2, decision layer,
  transliteration).
- **Day 3:** 2–3 final candidates. Keep **at least one in reserve** until the last hours.
- Never spend a submission on a change that has not improved the test-mix-weighted validation
  score.
- Never tune thresholds on the public LB. It is a subset, and the final ranking is private-only.
- **Operational note** `[INFERENCE]` from the environment date: the challenge window began
  2026-09-25 00:00 IST, and the dataset was unpacked locally at 10:28 the same day. If the team is
  competing now, run Phases 2–3 (harness + first valid submission) before anything else. This is
  an ordering decision only; the ideal design above is unchanged.

---

## PART 4 — PIPELINE COMPONENTS IN FULL DETAIL (L1–L9)

Unless stated otherwise, everything in this part is `[STRATEGY]`, justified by the `[DATA]` facts
cited inline. Each layer lists the **PS requirement it serves**.

### L1 — Data ingestion and validation

**Serves:** `[PS]` file format, open-set country, every S1 entity must be covered.

1. **Reader.**
   - `pyarrow.csv.read_csv(path, parse_options=ParseOptions(delimiter="\t"))` with every column
     typed as `string`, `strings_can_be_null=False` and `null_values=[]`.
   - In pandas, the equivalent is `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
     na_filter=False)`.
   - **Why:** pandas' default NA parsing would turn literal names or addresses such as `NA`,
     `null` or `None` into NaN. Literal `null`/`N/A`/`NULL` *inside* addresses are frequent
     (about 1.7% of S2/S3 `[DATA]`) and must be handled by the normaliser, not by the parser.
   - Keep default CSV quote handling. It correctly decodes the `"""…"` fields `[DATA]`.
2. **Schema assertions** (fail fast):
   - the exact header;
   - the ID prefix matches the file;
   - IDs are unique;
   - row count equals the line count minus one;
   - **country is logged as an open set**: unseen labels are *reported*, never rejected or
     remapped.
3. **Canonical storage:** Parquet partitioned by `split/source/country` with the columns
   `entity_id, src, name_raw, addr_raw, country, row_idx`. Reloading takes about a second, and each
   downstream stage reads only one country partition, which is what fits in 16 GB `[DATA]` +
   environment.
4. **Coverage guard:** from this layer on, carry the **ordered list of all test S1 IDs**. The final
   writer iterates over *that list*, not over predictions, so no entity can go missing.

### L2 — Preprocessing and normalisation

**Serves:** `[PS]` noise patterns; §6.6 observed noise.

Each record is turned into several **views**. Each view is used by different blocking passes and
features.

| View | Name side | Address side |
|---|---|---|
| raw-lower | NFKC + casefold | NFKC + casefold |
| `norm` | Accent-folded (Latin only), punctuation → space, `&`/`+` → `and` | Abbreviations canonicalised, null tokens removed |
| `core` | `norm` minus legal forms, honorifics, junk and generic appendices | Street + number + locality tokens only (state/region/country removed) |
| `translit` | Native Indic tokens → Latin (learned) | Native state/city tokens → Latin (learned) |
| `nospace` | `core` with spaces removed (for handles and websites) | — |
| structured | `legal_form` (canonical set), `dba_alias`, `acronym`, `has_handle`, `script` | `house_numbers`, `unit`, `street_tokens`, `city`, `state_canon`, `postal`, `landmark`, `is_empty` |

**Name normalisation, step by step** (every rule is motivated by an observed pattern, §6.6):

1. Unicode NFKC and casefold. Strip zero-width characters.
2. **Latin-only accent folding** (NFKD then drop combining marks **only when the base character is
   Latin**). This removes the injected accents (`Ínc`, `Rídge`, `Immôbiliere`). **Do not** strip
   combining marks from Indic scripts: their vowel signs are combining marks, and removing them
   destroys the word.
3. **Junk removal:**
   - leading and trailing `>>`, `<<`, `--`, `...`, `#`, `@`, `*`, `|`;
   - trailing `| www.…` segments (keep them as an alternate view);
   - brackets around a single token (`[Committee]`, `(Limited)`), keeping the content;
   - quote artefacts (`"ehpad Club`).
4. **DBA split:** on `trading as | t/a | a/k/a | aka | f/k/a | fka | formerly | d/b/a | dba`. The
   observed form is `<invented alias> <marker> <original name>`, so the **segment after the marker
   is the primary name** and the alias goes to `dba_alias`.
5. **Honorifics and prefixes** removed into a flag: `mr, mrs, ms, m/s, shri, sri, smt, the`. These
   are learned noise: `Sri Shree Entertainment`, `M/s JAGDAMBA`.
6. **Legal forms** extracted into a canonical set:
   - India and US: `pvt|private → PRIVATE`, `ltd|limited → LIMITED`, `llc|l.l.c. → LLC`,
     `inc|incorporated → INC`, `corp|corporation → CORP`, `co|company → CO`, `llp|l.l.p. → LLP`,
     `lp, pc, pllc, plc, public`;
   - France: `sarl|s.a.r.l.`, `sas|s.a.s.`, `sasu`, `eurl`, `sa`, `sci`, `snc|s.n.c.`, `sca`.

   This is a generic linguistic table (§1.13). Corrupted legal tokens (`PDRVIMTSE`, `Psravge`,
   `Prviae`) are caught by fuzzy matching against the legal vocabulary (edit distance ≤ 2 for
   tokens of length ≥ 5).
7. **Generic appendices:** the data appends tokens such as `Center`, `Services`, `Partners`,
   `Service`, `Group`, `Global`, `(India)`, `(France)` (`Krishna Finance Limited Center`,
   `Capital LLC Services`). **Learn this list from train pairs** as tokens that are significantly
   over-represented in S2/S3 relative to their matched S1. Drop them from `core` but keep them in
   `norm`.
8. **Handles and websites:** strip `@`, `#`, `.com/.in/.fr/.net/.org`, and a trailing `com`
   (`UrologypartnersCom`, `CORALIEFETESSASCOM`). Set `has_handle`. Match these through the
   `nospace` view (`capitalholding` = `capitalholding`). Optionally segment them with a
   corpus-frequency word list (dynamic programming); the dictionary is built only from the provided
   names.
9. **Acronym view:** initials of the `core` tokens, with and without legal forms (`Gallagher Crystal
   John LLC` → `gcj`, `gcjl`). Records whose whole name is 1–5 letters (`GCJ`, `EC`, `TC`, and the
   most frequent S2/S3 names `cc`, `pc`, `sc` `[DATA]`) get `is_acronym`.
10. **Transliteration** of Indic-script tokens (the most valuable learned component):
    - **Training signal** `[DATA]`: about 24% of matched India-S2 names and about 13% of India-S3
      names are native-script, and their S1 partner is Latin. That is roughly 0.5M aligned pairs,
      which is a large parallel corpus inside the provided data.
    - **Token dictionary:**
      - tokenise native text with `[\p{L}\p{M}\p{N}]+`, so vowel signs stay attached;
      - where the native and Latin token counts are equal, align them positionally;
      - otherwise align monotonically, maximising character-model scores;
      - count `native_token → latin_token` pairs and keep the argmax when its share is ≥ 0.6;
      - examples: `प्राइवेट → private`, `लिमिटेड → limited`, `फाइनेंस → finance`,
        `कृष्ण → krishna`.
    - **Character model for OOV tokens:** a small character-level model trained on the dictionary
      pairs. Options: a pair-HMM/WFST, a joint-sequence (grapheme→grapheme) model, or a tiny
      character transformer (a few million parameters, trained from scratch, so no licence
      issue).
    - **Rule fallback:** a deterministic Unicode-block transliteration (ISO-15919-like), for
      example the MIT-licensed `indic-transliteration` package or hand-written tables, followed by
      a **phonetic key** (collapse long vowels and geminates; `v/w`, `sh/s`, `ph/f`, `ee/i` and
      similar) so that near-miss transliterations still collide.
    - **Coverage audit:** the share of native tokens in the *test* S2/S3 that the dictionary covers
      (§6.11). Because 34% of test S1 lowercase names also occur in train S1 `[DATA]`, the name
      vocabulary is largely shared between train and test, and coverage is expected to be high
      `[INFERENCE]`.
    - **Leakage control:** when validating, learn the dictionary **on the training folds only**.
11. Output: all views plus flags such as `was_translit`, `translit_coverage`, `n_tokens_core` and
    `script`.

**Address normalisation, step by step:**

1. Casefold, NFKC, Latin-only accent folding. Remove tokens such as `null`, `n/a`, `na`, `none`
   and `-`, and set `null_count`.
2. **Remove PO Box/PMB and their numbers.** These are injected noise: `PMB 3330` and `PO BOX 4187`
   appear on records of the same entity `[DATA]`. Keep `has_pobox` as a flag.
3. **Street-type canonicalisation:**
   - US: `st, rd, dr, ave/av, ln, ct, cir, blvd, pkwy, hwy, trl, pl, ter, tpke, way`;
   - India: `marg, nagar, road, rd, colony, sector, phase, block`;
   - France: `r|r. → rue`, `bd|bd. → boulevard`, `av → avenue`, `all|all. → allee`,
     `imp → impasse`, `ch → chemin`, `pl → place`, `fg|fbg → faubourg`.

   **Beware `St → Saint`** `[DATA]` (`HAMILTON SAINT`): add a `saint ↔ st ↔ street` equivalence
   class instead of a single mapping.
4. **Ordinals:** `2nd|second|2st|2rd → 2`, `4th|fourth → 4`, `8th|eighth|egihth → 8`. Fuzzy-match
   number words with edit distance ≤ 2.
5. **House-number parsing:**
   - extract numeric tokens, stripping leading zeros (`008644 → 8644`);
   - split ranges (`2210-2214 → {2210, 2214}`, and treat the range as *containing* 2210);
   - split slash forms (`2/369`, `123/3`, `28/1`);
   - separate letter suffixes (`343A`, `7A`, `53 bis`, `17 G`);
   - remove prefixes `#`, `##`, `No.`, `N°`, `H.No`, `Hn`, `D/`, `C-`;
   - keep all numbers as a set, plus a `primary_number` (the first number on the street line).
6. **State and region canonicalisation**, learned from data:
   - for each country, find the address components that occupy the "state slot" (high-frequency
     trailing or leading components);
   - map variants to a canonical form by majority co-occurrence across matched train pairs
     (`TX ↔ Texas`, `MH ↔ Maharashtra`, `महाराष्ट्र ↔ Maharashtra`, `ಕರ್ನಾಟಕ ↔ Karnataka`,
     `தமிழ்நாடு ↔ Tamil Nadu`).
   - **France** (no labels): region and département tokens (`Nouvelle-Aquitaine` vs `Gironde`,
     `Hauts-de-France` vs `Nord`/`Pas-de-Calais`, `Pays de la Loire` vs `Loire-Atlantique`) are
     treated as **administrative tokens**. Down-weight them (their IDF is low anyway), or learn
     their equivalence *unsupervised* from co-occurrence with the same city tokens in test France
     S2/S3. **Do not type in an external region table** (§1.13).
7. **City:**
   - extract a city candidate: the component before the state, or a known-city token learned from
     frequency;
   - keep `city_of X ↔ X` and `X city ↔ X` equivalences;
   - note the **city substitution** `[DATA]` (Memphis ↔ Cordova, Rome ↔ Stout, Rotterdam ↔
     Schenectady). City disagreement must therefore be a *soft* feature, never a hard filter.
8. **Landmarks:** tag `near|nr|opp|opposite|behind|beside <phrase>` and keep the phrase as a
   separate low-weight view.
9. **Component-order invariance:** every address feature works on token sets or per-component
   matching, because reordering is pervasive (`OH, Columbus, 5559 Orville Avenue`).
10. `is_empty` for about 3% of S2/S3 records `[DATA]`. These rely entirely on name and context
    evidence.

**Unit tests:** every example in §6.6 becomes a test case (input → expected views).

### L3 — Blocking and candidate generation (produces `candidate_pairs.tsv`)

**Serves:** `[PS]` tip 1 (recall ceiling) and the `candidate_pairs.tsv` definition.

**Hard partition: country.** 100% agreement in train `[DATA]`. It shrinks the all-pairs space by
about 1.9× on train (two countries) and about 2.6× on test (three countries), computed from the
per-country row counts, and makes France its own independent problem.

Within each country and each target source (S2 and S3 separately; each S1 has about 1.67 S2 and
1.79 S3 matches on average `[DATA]`), run these passes:

| Pass | Representation | Retrieval | What it rescues |
|---|---|---|---|
| **P1 name-char** | `core` name char 3-grams (and 4-grams), TF-IDF, sublinear TF, IDF fit on S1 ∪ pool of that country | Top-K₁ cosine per S1 (K₁≈30) | Typos, suffix variation, reordering |
| **P2 address** | `core` address word tokens + char 3-grams of number+street, TF-IDF | Top-K₂ per S1 (K₂≈30) | Invented names, handles, acronyms, transliteration failures (§6.8: 8.0% of all matched pairs share no core name token despite Latin script on both sides) |
| **P3 name+locality** | `core` name + city + primary number | Top-K₃ (≈20) | Generic names disambiguated by locality |
| **P4 translit** | P1 on the `translit` view of native-script pool records | Top-K₄ (≈20) | Indic-script names |
| **P5 exact keys** | Group joins on (`core` sorted tokens), (`nospace`), (primary number + first street token), (acronym + city) | All pairs in blocks of size ≤ 1,000 (skip larger blocks) | Cheap, high-precision anchors |
| **P6 dense** | Multilingual embedding of `name \| address` | FAISS HNSW/IVF top-K₆ (≈20) | Semantic and cross-script variants, residual misses |
| **P7 reverse** | P1/P2 with the roles swapped: each pool record → top-k S1 (k≈3) | Add (S1, record) pairs | High-fan-out S1 entities whose top-K is saturated by look-alikes |

**Implementation at this scale:**

- Sparse top-K multiplication with `sparse_dot_topn` (MIT, multithreaded,
  `sp_matmul_topn(Q, P.T, top_n=K, threshold=t, n_threads=…)`), or chunked sparse×dense on CPU or
  GPU as in `amazon_ml_2026_analysis/eda/eda_block.py`. Chunk queries at about 100k.
- Memory for one country×source pool: about 2.4M rows × about 35 char-3-grams ≈ 85M nonzeros ≈
  0.7 GB (float32 values + int32 indices). This fits in 16 GB when partitions run sequentially
  `[INFERENCE]` (sized from the measured row counts).
- Dense pass: FAISS (MIT) HNSW with inner product on L2-normalised float16/float32 vectors. Use
  per-partition indices to bound memory (4.9M × 384 × 2 bytes ≈ 3.8 GB for the largest pool in
  float16; use IVF-PQ if memory is tight).
- **Mandatory correctness check (a lesson learned during this analysis `[DATA]`):** every fast
  retrieval implementation must be asserted against an **independent** exact brute force on a
  sample of queries, using the *unrestricted* matrices. Both the K-th best score and the
  score-at-returned-index must agree.
  - **What happened.** The first two runs of our own blocking experiment restricted the TF-IDF
    matrices to the "query vocabulary" by reading `.indices` from a **CSC** matrix. In CSC,
    `.indices` holds *row* indices, so the restriction silently selected feature IDs `0..n_queries`.
  - **How badly it misled.** Name-only recall@20 was reported as 18% against a brute-force truth
    of **79.8%**, with no error raised.
  - **Why the first self-check missed it.** It compared against brute force on the *same
    restricted* matrices, so it was not independent. A first hypothesis (unsorted CSR indices)
    was tested and **disproved**: identical numbers after that fix.
  - **Fix.** Derive columns from CSR, keep `sort_indices()` and
    `torch.sparse.check_sparse_tensor_invariants` as extra guards, and make the check independent.

**Union → pruning pre-ranker → final candidates:**

1. Union all passes and deduplicate. Keep per-pair provenance: which passes found it and its best
   rank in each pass.
2. The **pre-ranker** is a small LightGBM on cheap features: pass scores and ranks, name and
   address cosines, number overlap, and the source. It is trained on validation folds.
3. Keep the top N per (S1, source), with N chosen where the **oracle F0.5 given candidates** stops
   improving (for example, N = 15 per source).
4. **This pruned set is `candidate_pairs.tsv`.** The documentation states plainly that candidate
   generation = retrieval passes + pre-ranker, and that the main matcher scores exactly this set
   (§1.7).

**Blocking metrics** to report per country, source and pass:

- pair completeness (recall);
- entity-level recall ceiling (the oracle macro F0.5);
- reduction ratio = 1 − |candidates| / (|S1_c| × |pool_c|);
- mean and 95th-percentile candidates per S1.

The measured baseline for P1+P2 alone is in §6.9.

**Why not other classic blockers:**

- **Sorted-neighbourhood** keys break under prefix junk (`>> Dss Care`), word reordering
  (`Edge Ulhashnagar Ltd`) and native scripts `[DATA]`.
- **Standard blocking** on PIN/ZIP fails because codes are present in fewer than 7% of addresses
  `[DATA]`.
- **MinHash-LSH** is viable for scale, but gives weaker control over the per-query budget than
  top-K TF-IDF. Keep it as an alternative if the sparse multiplication becomes the bottleneck.

### L4 — Feature engineering

**Serves:** `[PS]` tip 2 (string similarity); country-specific patterns.

About 80 features per (S1, candidate) pair, computed vectorised with `rapidfuzz.process.cpdist`
and numpy across worker processes. Estimated cost: about 50M test pairs × about 25 string
comparisons at about 1–2 µs ≈ 20–40 CPU-minutes, spread over 12 threads `[INFERENCE]`.

| Group | Features | Notes |
|---|---|---|
| **N — name** | rapidfuzz `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, `WRatio`, Jaro–Winkler, normalised Levenshtein, each on `norm`, `core` and `translit`; char-3-gram TF-IDF cosine; word TF-IDF cosine; Jaccard and overlap coefficient on `core` tokens; **IDF-weighted soft-Jaccard / Monge–Elkan(JW)**; `nospace` equality and containment; acronym match (both directions); legal-form relation (equal / compatible / one missing / conflicting); DBA present plus alias similarity; handle flag; script flags and `translit_coverage`; numbers-in-name agreement; first-token and last-token equality; length and token-count differences; count of generic appendix tokens | The IDF weighting captures *rarity* of the agreement. Two records sharing "Lifeco" is strong evidence; sharing "care" is weak. |
| **A — address** | `token_set_ratio`, `token_sort_ratio`, `partial_ratio`; word and char TF-IDF cosines; **number features**: exact set equality, primary-number equality, any overlap, *conflict* (both sides have numbers and none overlap), minimum digit-edit distance (catches `3280` vs `328`), range containment; street-token similarity without numbers or types; city equal, alias or missing; state-canonical equal; postal code equal, conflict or missing; unit/flat match; landmark flag and phrase similarity; empty and null flags on each side; component-count difference | Number *conflict* is one of the strongest negative signals. Matched pairs share a number 87–97% of the time versus 0.2–8.7% at random `[DATA]`. |
| **F — specificity/frequency** (unsupervised, fit on the scored pool) | Frequency of the S1 `core` name among S1 entities of the country (the generic-name indicator); pool name frequency; **address frequency among S1** (shared buildings); summed IDF of shared tokens | Tells the model when a name match is cheap evidence (`bordeaux club sarl` ×205 `[DATA]`) |
| **C — context/competition** | Rank of the candidate among the S1's candidates (per source and overall) by pre-ranker score; score gap to that S1's best; number of candidates above fixed score levels; **rank of the S1 among the record's own S1 candidates; is this S1 the record's best; margin to the record's second-best S1**; **sibling support** (count and mean similarity of the S1's other candidates that are highly similar to this record); pass-provenance bitmask; best per-pass rank | Encodes the exclusive-assignment structure (§0.4 item 3) directly into the model. This is typically among the strongest feature groups in ER. |
| **S — semantic** | Embedding cosine (name, address, combined); cross-encoder probability (optional, **OOF-stacked**) | Multilingual models bridge scripts and paraphrase |
| **M — meta** | Source (S2/S3); whether both sources are present among the candidates | **No country one-hot.** Country-agnostic modelling is required for France (validated by LOCO, §4.7). |

**Leakage and validation notes per group:**

- Never use `entity_id` digits or row positions. They are verified uninformative `[DATA]`, and a
  model could still latch onto spurious patterns.
- F- and C-group features use **only unlabeled candidate-graph information**. Compute them on
  exactly the pool being scored: the validation pool for validation and the test pool for test.
- Anything learned from labels (transliteration dictionary, appendix-token list, state map,
  pre-ranker, cross-encoder) must be learned on **training folds only** when producing validation
  scores. Stacked inputs must be out-of-fold.
- Fit IDF/TF-IDF on the pool being scored (transductive, label-free). Document this (§1.13 grey
  area (i)). Fitting on train only would leave French vocabulary without IDF.

### L5 — Matching / scoring model

**Serves:** `[PS]` ML solution; licence constraint 5.

1. **Primary: LightGBM binary classifier** (MIT).
   - **Why a GBDT fits this data:** heterogeneous similarity features with non-linear interactions
     (a generic name *and* an address match → match; a strong name *and* a number conflict →
     non-match); robust to missing values (empty addresses); fast on tens of millions of rows;
     trivially within the licence and size rules.
   - **Training rows:**
     - all blocking candidates of the training-fold S1 entities;
     - positives are GT pairs inside the candidates;
     - negatives are the remaining candidates. These are *realistic hard negatives*, far harder
       than random pairs (§6.8).
   - **Budget:** 1.77M training S1 × about 30 candidates ≈ 53M rows × 80 float32 features ≈ 17 GB.
     So either subsample S1 entities (keeping each entity's full candidate list, because the C
     features depend on it) down to about 10–15M rows on the laptop, or train on a
     memory-optimised instance (§5.3). Check a learning curve before paying for scale.
   - **Starting hyperparameters:** `num_leaves 127–255`, `learning_rate 0.05`,
     `min_data_in_leaf 200`, `feature_fraction 0.8`, `bagging_fraction 0.8`, `lambda_l2 1`,
     early stopping on fold validation (AUC-PR and log-loss). Tune with Optuna on a fixed
     subsample.
   - **Monotone constraints** on a handful of core similarities (name and address similarity ↑)
     improve zero-shot robustness for France. Keep them only if LOCO improves.
2. **Diversity models** for an ensemble: XGBoost (Apache-2.0) or CatBoost (Apache-2.0) on the same
   features. Average in logit space after calibration.
3. **Cross-encoder** (optional, strongly useful for script-mixed and heavily corrupted pairs):
   - fine-tune a multilingual encoder on `"name | address"` pairs;
   - candidates: `microsoft/mdeberta-v3-base` (MIT, about 280M) or `FacebookAI/xlm-roberta-base`
     (MIT, about 278M);
   - train on about 2–5M hard pairs drawn from blocking outputs;
   - run inference **only on the uncertain band** (for example 0.05 < p < 0.95 after LightGBM),
     which keeps the cost bounded;
   - feed its OOF probability to a second-stage LightGBM (stacking).
4. **Bi-encoder for P6 blocking** (optional): fine-tune `intfloat/multilingual-e5-small` (MIT,
   about 118M) with in-batch negatives on train pairs. This raises P6 recall.
5. **LLM** (optional, low priority): at most 8B parameters, Apache/MIT, **run locally**. Examples
   are `Qwen2.5-7B-Instruct` (Apache-2.0) and `Mistral-7B-Instruct-v0.3` (Apache-2.0). The only
   role is adjudicating a few thousand borderline pairs. Its expected gain per unit cost is low
   next to the cross-encoder. **Never** use an LLM to generate synthetic "French data": that sits
   too close to "external data augmentation".

**Licence register (verify every row on the model card at download time, and pin revisions):**

| Component | Licence | Params | Role |
|---|---|---|---|
| LightGBM | MIT | n/a (trees) | Primary matcher, pre-ranker |
| XGBoost / CatBoost | Apache-2.0 | n/a | Ensemble diversity |
| `intfloat/multilingual-e5-small` / `-base` | MIT | ≈118M / ≈278M | Embeddings (P6, S-features) |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | ≈118M | Embedding alternative |
| `sentence-transformers/LaBSE` | Apache-2.0 | ≈471M | Cross-script embedding alternative |
| `BAAI/bge-m3` | MIT | ≈568M | Stronger multilingual embedding alternative |
| `microsoft/mdeberta-v3-base` | MIT | ≈280M | Cross-encoder |
| `FacebookAI/xlm-roberta-base` | MIT | ≈278M | Cross-encoder alternative |
| `google/muril-base-cased` | Apache-2.0 | ≈237M | Indic-focused encoder alternative |
| Qwen2.5-7B-Instruct / Mistral-7B-Instruct-v0.3 | Apache-2.0 | ≈7.6B / ≈7.3B | Optional adjudicator. **Would consume almost the whole 8B budget if the total is counted.** |
| **Avoid** | Llama/Gemma community licences, GPL tools (for example `Unidecode` is GPL-2.0; use `anyascii` (ISC) or our own tables) | — | Licence hygiene |

Parameter counts are approximate `[INFERENCE]`. Licences are as generally published; **verify
them** `[BEST-PRACTICE]`.

### L6 — Decision, thresholding and consistency

**Serves:** `[PS]` F0.5 precision emphasis, singletons, one-to-many output.

1. **Calibration.** Isotonic regression (or Platt) on OOF scores, fitted per source (S2/S3). Check
   a reliability diagram per country. Calibrated p is required for steps 3–4.
2. **Exclusive assignment** `[DATA]`-justified: every S2/S3 record maps to at most one S1.
   - For each pool record r, keep only `argmax_s1 p(s1, r)`.
   - Optionally drop *both* candidate links when the top two S1 entities are near-tied and both are
     high (an ambiguity guard). Tune the margin on OOF.
   - **Why it helps precision:** if the model links r to two S1 entities, at least one link is
     certainly a false positive, costing four false negatives' worth of F in that entity's
     denominator.
3. **Rank-dependent thresholds** (the simple version; derivation in §1.11):
   - sort each S1's surviving candidates by p;
   - accept rank 1 if p₁ ≥ t₁, rank 2 if p₂ ≥ t₂, rank 3 and beyond if pₖ ≥ t₃;
   - grid-search t₁ ≤ t₂ ≤ t₃ on OOF.
   - The math predicts t₁ ≪ t₃ (roughly 0.1 vs 0.75 for a 4-match entity).
4. **Expected-F0.5 set selection** (the principled version).
   - For an S1 with calibrated candidates p₁ ≥ … ≥ pₘ, choose the k ∈ {0, …, m} that maximises
     `E[F0.5(top-k)]`.
   - Model the outcomes as yᵢ ~ Bernoulli(pᵢ), plus a count M of true matches *missed by
     blocking*: Poisson with mean λ equal to the validation-measured miss rate, per country and
     source.
   - Then `F = 1.25·TP / (1.25·TP + 0.25·(FN_cand + M) + FP)`, with F = 1 when k = 0 and there
     are no true matches at all.
   - Estimate it with vectorised Monte Carlo (about 256 draws per entity, batched in numpy) or the
     plug-in approximation.
   - Handles singletons naturally (the k = 0 option) and adapts thresholds to each entity's
     evidence.
   - Keep whichever of (3) or (4) validates better.
5. **Graph consistency** (optional).
   - Build S2↔S3 (and S2↔S2, S3↔S3) sibling edges among the candidates of each S1. Use high
     name+address similarity, or an **auxiliary sibling model** trained on GT co-cluster pairs,
     which are records sharing an S1 in train (labels from the provided data).
   - Records that are strongly siblings of confidently-assigned records get a boost. Conflicting
     sibling evidence lowers confidence.
   - Implement it as a *second-stage* model with sibling-support features, not as an ad-hoc rule,
     so it stays calibrated.
6. **Final guarantee:** the final sets are ⊆ `candidate_pairs` by construction. The writer asserts
   it.

### L7 — Validation and self-scoring

**Serves:** `[PS]` "hold out a validation split … score it yourself".

1. **Exact scorer** (reference implementation; unit-test it with the PS example = 0.714):

   ```python
   def f05_entity(pred: set, true: set) -> float:
       if not true:                      # singleton: only an empty prediction scores
           return 1.0 if not pred else 0.0
       tp = len(pred & true)
       if tp == 0:
           return 0.0
       fp, fn = len(pred) - tp, len(true) - tp
       return 1.25 * tp / (1.25 * tp + 0.25 * fn + fp)

   def macro_f05(pred_map: dict, true_map: dict) -> float:
       return sum(f05_entity(pred_map.get(k, set()), v) for k, v in true_map.items()) / len(true_map)
   ```

2. **Folds.** 5 folds over S1 entities, stratified by country × match-count bucket (0, 1, 2, 3,
   4, 5, 6+). Because clusters are disjoint `[DATA]`, the S1 grouping *is* the cluster grouping,
   so there is no cross-fold pair leakage.
3. **Pool construction per fold.** The pool is the fold's matched S2/S3 plus distractors.
   - **Train-like density:** a random country-matched share of the unmatched records, sized so
     distractors make up 26.6% of the S2 pool and 25.4% of the S3 pool `[DATA]`.
   - **Test-like density (important):** test has about 25% more S2/S3 per S1 in every country
     `[DATA]`. If fan-out is unchanged, distractors make up about 41% of test pools
     `[INFERENCE]`. Build a second validation variant with that density by adding **matched
     records of *other folds'* S1 entities**. This is realistic: they are records of real
     businesses whose S1 is absent, which is plausibly how test distractors arise.
   - Select thresholds that are robust across **both** densities.
4. **Test-mix weighting.** Report F0.5 per country, plus `0.4675·India + 0.3827·US + 0.1497·France`,
   using the test S1 mix `[DATA]`. France is proxied by the LOCO estimate (next item).
5. **Leave-one-country-out (LOCO) as the France proxy.** Train everything label-dependent on US
   only and evaluate on India, and vice versa.
   - The degradation versus in-country training estimates the zero-shot penalty.
   - Use LOCO to choose between feature sets, monotone constraints, and transductive or inductive
     IDF.
   - India→US is a *harsher* shift than →France (France shares the Latin script with US), so this
     proxy is conservative `[INFERENCE]`.
6. **Uncertainty.** Bootstrap over S1 entities for 95% intervals. With about 440k validation
   entities per fold, CI half-widths are tiny (±0.001-level) `[INFERENCE]`. Differences below
   about 0.002 are noise from threshold tuning, not real gains.
7. **Blocking diagnostics** alongside every run: pair completeness, oracle F0.5, candidates per S1
   (§L3).
8. **LB tracking.** Record validation (both densities, test-mix weighted) against the public LB in
   the ledger. A stable offset is fine; a changing offset signals distribution-specific overfit.

### L8 — Output formatting and submission validation

**Serves:** `[PS]` output format and constraints 1–4.

1. **Writer** (LF-only, exact header, iterates over the full ordered test S1 list):

   ```python
   def write_id_lists(path, second_col, s1_ids_in_order, mapping):
       with open(path, "w", encoding="utf-8", newline="\n") as f:        # never "\r\n"
           f.write(f"source1_entity_id\t{second_col}\n")
           for s1 in s1_ids_in_order:
               ids = list(dict.fromkeys(mapping.get(s1, ())))           # dedupe, keep order
               f.write(s1 + "\t" + ",".join(ids) + "\n")

   write_id_lists("output/candidate_pairs.tsv", "candidate_entity_ids", test_s1_ids, cands)
   write_id_lists("output/matching_results.tsv", "matched_entity_ids", test_s1_ids,
                  {k: [i for i in v if i in cands_set[k]] for k, v in matches.items()})
   ```

2. **Pre-upload assertions**, all in code:
   - row count = 1,732,544 `[DATA]`;
   - set of S1 IDs = the test S1 IDs;
   - `b"\r" not in bytes`;
   - valid UTF-8;
   - every ID has an `S2-`/`S3-` prefix and **exists in the test pool**;
   - no intra-list duplicates;
   - matches ⊆ candidates;
   - no spaces or quotes inside the ID column.
3. **Official validator** with `--check-ids`, in two runs: matching only (lower memory), then with
   `--candidate`. Upload only on `PASS`.
4. **Sanity statistics** logged per submission: fraction of empty rows, mean list length per
   country, and the S2/S3 split. Compare them to validation expectations. For example, if the
   predicted empty rate is far above about 6%, something is broken.

### L9 — Reproducibility and documentation

**Serves:** `[PS]` final package; `[GUIDE]` version history and commented code.

1. **Repository layout:**

   ```
   code/business_entity_resolution/
   ├── README.md                      # exact commands + hardware + runtimes
   ├── requirements.txt               # pinned (pip freeze of the run environment)
   ├── configs/final.yaml             # every threshold, K, N, seed, model revision
   ├── src/ber/
   │   ├── io.py            (L1)      ├── normalize/{names,addresses,translit,states}.py (L2)
   │   ├── blocking/{tfidf,keys,dense,prerank}.py (L3)
   │   ├── features/{name,address,freq,context,semantic}.py (L4)
   │   ├── models/{gbdt,cross_encoder,calibrate}.py (L5)
   │   ├── decide/{assign,setselect,graph}.py (L6)
   │   ├── evaluate/{scorer,folds}.py (L7)   └── output/writer.py (L8)
   │   └── run.py                    # python -m ber.run --data ../../dataset --out ../../output
   └── tests/                         # scorer, normalisers, writer
   ```

2. **Determinism:** fixed seeds for numpy, LightGBM and torch; deterministic sort orders; cached
   stage outputs keyed by config hash.
3. **The README** gives setup (`python -m venv`, `pip install -r requirements.txt`), the data
   location, one end-to-end command, per-stage commands, expected runtime and peak RAM per stage,
   how pretrained weights are obtained (pinned model IDs and revisions; offline cache supported),
   and the validator command.
4. **Mapping to `Documentation_template.md`:**

   | Template section | Content |
   |---|---|
   | §2.1 Problem Analysis | Part 6 findings |
   | §2.2 Approach | "Blocking + GBDT + constrained decision (hybrid)"; core innovations: learned transliteration, exclusive assignment + expected-F0.5 selection, zero-shot France design |
   | §3 Blocking | Keys and passes, candidate count, pair-completeness table |
   | §4 Matching model | Features, model, threshold method |
   | §5 Results | Validation F0.5 per country and test-mix weighted, LB scores, FP/FN taxonomy |
   | Appendix A | Code structure |
   | Appendix B | Ablation tables and reliability plots |

5. **Every comment and docstring states exactly what the code does.** No overclaiming: for example,
   call transliteration "learned from training pairs", not "language understanding".

---

## PART 5 — TECHNOLOGY STACK (AWS-PRIORITISED) AND AWS OPERATIONS

### 5.1 Principles

- The deliverable is **two static TSV files plus code and docs**, produced by an **offline batch**
  pipeline. Nothing is served.
- AWS services are included **only where they do a job the pipeline needs** (storage, elastic
  compute beyond the laptop, logging, access control, cost control).
- Production-serving infrastructure is **deliberately excluded**. The AWS-FAQ itself says an
  endpoint is unnecessary ("you only need to submit a CSV, not host an endpoint") `[AWS-FAQ]`, and
  the workshop says "Notebook Instance + local training + local prediction. You submit a CSV, not
  a running API" `[WORKSHOP]`.

### 5.2 AWS services: inclusion and exclusion with justification

| Service | Decision | Layer | Why it is needed / why it beats the alternative | Advantage |
|---|---|---|---|---|
| **Amazon S3** | **Include** | Data and artefact bus | Single source of truth for raw TSVs, Parquet partitions, blocking/feature artefacts, models and every output version. It is also the documented cross-account sharing mechanism `[AWS-FAQ]`. **Bucket versioning** gives the submission history `[GUIDE]` for free. | Durability; team sharing; reproducibility |
| **SageMaker Notebook Instance** | **Include (control plane only)** | Development | Faster to start than Studio (no Domain) `[WORKSHOP]`. **But `ml.t3.medium` has 4 GiB of RAM**, which cannot hold even one full source file in memory at this scale (5M rows; §6.1). Use it for code, orchestration and *sampled* experiments only. | Free-tier hours `[AWS-FAQ]` |
| **SageMaker Processing Jobs** | **Include** | L1–L4 at full scale (normalisation, blocking, features, test inference) | Ephemeral, right-sized instances (memory-optimised for sparse retrieval; GPU for embeddings) that **stop automatically**, so there is no idle cost. Stdout goes to CloudWatch; inputs and outputs go to S3. Beats a long-lived EC2 box on cost safety and reproducibility (the job definition *is* the run record). | Scalability; cost control |
| **SageMaker Training Jobs** | **Include** | L5 (LightGBM at scale; cross-encoder fine-tuning) | The free tier covers 50 h of `ml.m5.xlarge`/`m4.xlarge` `[AWS-FAQ]`, which is enough for sub-sampled GBDT training. GPU instances (`g4dn`/`g5`) handle transformer fine-tuning. **Managed Spot Training** with S3 checkpoints cuts credit burn. The output `model.tar.gz` lands in S3 and is shareable across accounts `[AWS-FAQ]`. | Elastic GPU; spot savings |
| **Amazon CloudWatch** (+ **AWS Budgets** alarms) | **Include** | Operations | Job logs and metrics arrive automatically. Billing alarms are the workshop's first operational instruction `[WORKSHOP]`. | Visibility; safety |
| **AWS IAM** | **Include** | Security | A SageMaker execution role scoped to `s3://<bucket>/*` and SageMaker actions; cross-account read via the bucket policy `[AWS-FAQ]`. | Least privilege |
| **Amazon ECR** | Optional | Reproducibility | A custom image with pinned `requirements.txt` makes every job identical. Alternatively, use the prebuilt SageMaker scikit-learn/PyTorch containers and `pip install -r` at job start. | Environment parity (the AWS-FAQ "same container" caveat) |
| SageMaker Batch Transform | Exclude (use Processing) | — | Designed for model-container inference. A Processing job running our own script (with a GPU for embeddings) is simpler for this pipeline. | — |
| SageMaker **Endpoints**, API Gateway, ECS/EKS, Lambda | **Exclude** | — | Nothing is served. Endpoints bill while idle `[WORKSHOP]` and cannot be shared cross-account without Organizations `[AWS-FAQ]`. Lambda's 15-minute and 10 GB limits suit no stage of this pipeline. | Avoids waste |
| AWS Glue, Amazon Athena | Exclude | — | 2.5 GB of TSV is handled in seconds by pyarrow locally (full-file read in 0.3–1.3 s `[DATA]`). Spark or serverless SQL adds cost and complexity for no gain. Athena could serve ad-hoc SQL EDA, but that EDA is already done. | — |
| Amazon Bedrock | **Exclude** | — | Hosted foundation models are mostly neither MIT/Apache nor ≤8B (constraint 5 `[PS]`). A hosted API call also invites fair-play questions (§1.13). Local ≤8B models cover every legitimate LLM use. | Compliance |
| DynamoDB, RDS, Redshift | Exclude | — | No online state and no relational serving need. Parquet on S3 is the feature store. | — |
| EC2 (direct) | Alternative only | — | Viable for a GPU or high-memory box. SageMaker jobs are preferred for the free-tier hours and automatic shutdown. Free-tier EC2 micro instances (about 1 GiB) are useless at this scale `[INFERENCE]`. | — |
| Custom KMS keys | **Avoid** | — | They break cross-account decryption `[AWS-FAQ]`. | — |

### 5.3 AWS architecture and operational lifecycle, end to end

1. **Accounts:** one per member, **all in the same region** (for example `us-east-1` `[WORKSHOP]`),
   because cross-region copies consume credits `[AWS-FAQ]`.
2. **Billing guardrails first:** AWS Budgets alarms at, say, 25%, 50% and 80% of credits, plus
   Billing → Free Tier monitoring `[AWS-FAQ]`.
3. **IAM:** a SageMaker execution role with `s3:GetObject/PutObject/ListBucket` on the team bucket
   only, plus `sagemaker:*Job*` and `logs:*`. For teammates, use the bucket policy from the
   AWS-FAQ (`arn:aws:iam::<ID>:root`, `GetObject` + `ListBucket`) or presigned URLs.
4. **S3 layout** (versioning on, default encryption):

   ```
   s3://hackathon-<team>/raw/{train,test}/*.tsv
   s3://hackathon-<team>/parquet/split=<train|test>/source=<1|2|3>/country=<...>/part-*.parquet
   s3://hackathon-<team>/artifacts/<stage>/<run_id>/...      # dictionaries, TF-IDF vocab, candidates, features
   s3://hackathon-<team>/models/<run_id>/model.tar.gz
   s3://hackathon-<team>/outputs/<run_id>/{matching_results.tsv,candidate_pairs.tsv}
   s3://hackathon-<team>/ledger/submissions.csv
   ```

   Mind the **free-tier limits** of 5 GB storage and 2,000 PUT per month `[AWS-FAQ]`. The raw data
   alone is 2.5 GB, so write **few large Parquet files**, not thousands of small ones, and expire
   intermediate artefacts.
5. **Notebook Instance vs Studio:** a Notebook Instance (faster, no Domain) as the control plane
   `[WORKSHOP]`.
6. **Local vs job, decided by the measured scale** `[DATA]` + `[INFERENCE]`:

   | Stage | Working set | Where |
   |---|---|---|
   | Sampled EDA and prototyping | < 1 GB | Laptop or `ml.t3.medium` |
   | Normalisation (about 24M records, train + test) | 2–4 GB per partition | Laptop (12 threads) or a Processing job, CPU, 16 vCPU / 64 GiB |
   | Blocking, per country × source | 1–4 GB per partition; wall time is the issue | Laptop sequentially (hours), or a memory-optimised Processing job (16 vCPU / 128 GiB) to run partitions in parallel |
   | Features (about 50M test pairs + training pairs) | Chunked | Laptop or Processing job |
   | LightGBM on 10–15M rows | 5–8 GB | Laptop is borderline; the free-tier `ml.m5.xlarge` (16 GiB) is similar; `ml.r5.2xlarge`/`4xlarge` is comfortable |
   | Embeddings (about 24M strings) | GPU | Local RTX 3050 **after installing a CUDA build of torch**, or `ml.g4dn.xlarge` (T4 16 GB) |
   | Cross-encoder fine-tuning | GPU | `ml.g5.xlarge` (A10G 24 GB) or `g4dn` |

   Instance specifications are general AWS knowledge `[BEST-PRACTICE]`:

   | Instance | vCPU / RAM | GPU |
   |---|---|---|
   | `t3.medium` | 2 / 4 GiB | — |
   | `m5.xlarge` | 4 / 16 GiB | — |
   | `m5.4xlarge` | 16 / 64 GiB | — |
   | `r5.4xlarge` | 16 / 128 GiB | — |
   | `g4dn.xlarge` | 4 / 16 GiB | 1× T4 |
   | `g5.xlarge` | 4 / 16 GiB | 1× A10G |

   Hourly prices are `[UNSPECIFIED]` here: check the SageMaker pricing page before launching, and
   set `MaxRuntimeInSeconds` on every job.

   **Honest note:** the free-tier SageMaker instances are *weaker* than the local laptop
   (12 threads, 16 GB, RTX 3050 6 GB). Free-tier hours are useful for parallel experiments across
   teammates. Credits pay for the few memory- or GPU-heavy runs.
7. **Artefacts:** every job writes to `artifacts/<stage>/<run_id>/`. The `run_id` = git SHA +
   config hash, which ties the S3 contents to the ledger.
8. **Cross-account credit pooling** `[AWS-FAQ]`:
   - share artefacts (dictionaries, candidates, models) via the bucket policy or presigned URLs;
   - teammates continue in their own accounts using the **same container and framework versions**;
   - jobs themselves cannot move.
9. **Logging:** CloudWatch Logs from jobs, plus the local ledger CSV. The ledger is the auditable
   version history `[GUIDE]`. SageMaker-managed MLflow is optional and costs money; the CSV ledger
   suffices `[STRATEGY]`.
10. **Cleanup:**
    - stop (don't delete) notebooks daily `[WORKSHOP]`;
    - no endpoints ever;
    - after the competition, delete notebooks, EBS volumes, ECR images and buckets (after
      archiving the final zip);
    - redeem top-500 credit codes via Billing → Credits `[AWS-FAQ]`.
11. **No hosted endpoint is required.**
    - Workshop guidance: "submit a CSV, not a running API" `[WORKSHOP]`.
    - AWS-FAQ: endpoints are not shareable cross-account and this is "not a problem here since you
      only need to submit a CSV" `[AWS-FAQ]`.
    - The PS deliverable is two TSV files `[PS]`.

### 5.4 Non-AWS stack

| Tool | Licence | What / why | Notes |
|---|---|---|---|
| Python 3.11/3.12 | PSF | Runtime | Local: 3.12.3 `[DATA]` (env) |
| **pyarrow** | Apache-2.0 | Fast multithreaded TSV/Parquet I/O; compact strings; vectorised string compute | Local 23.0.1. Full 500 MB file read in about 1 s `[DATA]`. |
| pandas | BSD-3 | Convenience frames | Local 3.0.2 (pyarrow-backed strings by default). **Always pass `keep_default_na=False`.** |
| DuckDB / Polars | MIT | Optional out-of-core joins (exact-key blocking, GT explode) | Not installed |
| numpy / scipy | BSD | Sparse matrices, numerics | Installed |
| scikit-learn | BSD-3 | `TfidfVectorizer`, isotonic calibration, splits | 1.8.0 installed |
| **sparse_dot_topn** | MIT | Multithreaded sparse top-K cosine: the blocking workhorse | Not installed |
| **RapidFuzz** | MIT | C++ string similarities (`cpdist` for pairwise batches) | 3.14.6 installed |
| jellyfish | MIT | Phonetic codes (for Latin names; transliterated names use our phonetic key) | Optional |
| **FAISS** | MIT | ANN for the dense blocking pass | `faiss-cpu` 1.13.2 installed |
| **LightGBM** | MIT | Primary matcher and pre-ranker | Not installed |
| XGBoost / CatBoost | Apache-2.0 | Ensemble diversity | Optional |
| Optuna | MIT | Hyperparameter search | Optional |
| PyTorch (**CUDA build**) | BSD-3 | Embeddings and cross-encoder | **Installed build is CPU-only (`2.11.0+cpu`)** `[DATA]` (env) |
| transformers / sentence-transformers | Apache-2.0 | Model loading and fine-tuning | transformers 5.16.1 installed |
| networkx | BSD-3 | Sibling-graph consistency (small per-S1 subgraphs) | 3.6.1 installed |
| indic-transliteration / anyascii | MIT / ISC | Rule-based transliteration fallback (**not** GPL `Unidecode`) | Optional |
| pytest, git | MIT, GPL (tool) | Tests; versioning, tags per submission | — |

**Experiment tracking:** a CSV/JSON ledger plus git tags is sufficient and fully auditable. MLflow
(Apache-2.0) is justified only if more than about 50 runs need comparison across members.

---

## PART 6 — DATASET INTELLIGENCE / DATA AUDIT (MEASURED)

> **Status flag required by the master prompt §5:** the master prompt expected the raw files to be
> missing. **They are present and were fully scanned.** Everything tagged `[DATA]` below is
> measured, not assumed. Scripts and exact commands are in Appendix A. Items that remain
> *unmeasured* are listed in §6.11 as an executable checklist. No statistic here is estimated
> unless it is explicitly tagged `[INFERENCE]`.

### 6.1 Files, sizes and parsing facts `[DATA]`

| File | Bytes | Data rows | Notes |
|---|---:|---:|---|
| `train/train_source1.tsv` | 210,069,713 | 2,206,821 | 4 escaped-quote lines |
| `train/train_source2.tsv` | 489,301,488 | 5,034,616 | 6 escaped-quote lines |
| `train/train_source3.tsv` | 503,705,637 | 5,285,603 | 0 |
| `train/train_ground_truth.tsv` | 127,015,583 | 2,206,821 | 2 columns |
| `test/test_source1.tsv` | 175,022,086 | **1,732,544** | 134 escaped-quote lines |
| `test/test_source2.tsv` | 509,456,422 | 4,887,273 | 349 |
| `test/test_source3.tsv` | 506,002,772 | 5,082,316 | 330 |

- Every file uses **CRLF** line endings. Every row has exactly 4 (or 2) tab-separated fields.
  There is no BOM, and the encoding is UTF-8.
- Pandas default parsing and `QUOTE_NONE` parsing give identical row counts. The default correctly
  unescapes `"""x"` → `"x`.
- Entity IDs: the numeric part has variable length (IDs are 4–12 characters in total), and IDs
  are unique within each file.
- A full-file read with pyarrow takes 0.3–1.3 s per file on the local machine (12 threads).

### 6.2 Per-source profile `[DATA]`

Values are percentages of rows. The regex definitions are in `eda_profile.py`.

| Metric | tr S1 | tr S2 | tr S3 | te S1 | te S2 | te S3 |
|---|---:|---:|---:|---:|---:|---:|
| Empty address | 0.00 | 3.36 | 3.33 | 0.00 | 2.65 | 2.68 |
| Address contains a `null` token | 0.00 | 1.75 | 1.65 | 0.00 | 1.44 | 1.39 |
| Name has Devanagari | 0.00 | 5.35 | 2.99 | 0.00 | 6.32 | 3.56 |
| Name has other Indic script (Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) | 0.00 | 4.07 | 2.28 | 0.00 | 4.86 | 2.75 |
| Address has Devanagari / other Indic | 0 / 0 | 5.51 / 3.97 | 5.22 / 3.78 | 0 / 0 | 6.51 / 4.75 | 6.28 / 4.55 |
| Name has Latin accents | 0.00 | 5.77 | 6.21 | 2.35 | 7.81 | 8.20 |
| Name all uppercase | 0.00 | **18.90** | 2.96 | 0.00 | **17.49** | 3.20 |
| Address all uppercase | 0.00 | **63.38** | 0.02 | 0.00 | **50.17** | 0.02 |
| Name looks like a URL/handle (`www.`, `.com`, `.in`, `.fr`) | 0.00 | 4.00 | 3.99 | 0.00 | 3.19 | 3.23 |
| Name has junk marks (`<<`, `>>`, `--`, `\|`, `*`, `#`) | 0.06 | 1.91 | 1.90 | 0.04 | 1.47 | 1.49 |
| Name has brackets | 2.06 | 7.72 | 8.01 | 3.61 | 8.34 | 8.68 |
| Name has a DBA / aka / f/k/a / trading-as marker | 0.00 | 0.00 | **1.39** | 0.00 | 0.00 | **1.05** |
| Address has a 6-digit number (PIN-like) | **0.08** | 0.84 | 0.81 | **0.06** | 0.56 | 0.54 |
| Address has a 5-digit number (ZIP / French postcode-like) | **6.69** | 6.72 | 6.71 | **4.38** | 4.68 | 4.66 |
| Address has PO Box | 0.00 | 1.00 | 0.94 | 0.00 | 0.65 | 0.61 |
| Address has "near/opp/behind…" | 5.31 | 4.54 | 3.45 | 6.19 | 5.39 | 4.05 |
| Address has Unit/Apt/Suite | 8.84 | 0.65 | 4.27 | 6.04 | 0.77 | 3.05 |

**Reading:**

- **S1 is the clean reference.** It has no native scripts, no all-caps, no handles and no empty
  addresses.
- **S2's signature** is all-caps (addresses in 50–63%), heavy native script and no DBA markers.
- **S3's signature** is DBA/aka markers, full US state names (`Texas`), Indian state *codes*
  (`MH`, `DL`, `KA`, `WB`, `GJ`) and native-script state names.
- **S1 writes** 2-letter US state codes and full Indian state names.

Source identity is therefore an informative feature for normalisation and modelling.

### 6.3 Train → test shift `[DATA]`

| | Train S1 | Test S1 | Train S2/S1 | Test S2/S1 | Train S3/S1 | Test S3/S1 |
|---|---:|---:|---:|---:|---:|---:|
| US | 1,323,633 (59.98%) | 663,106 (38.27%) | 2.279 | **2.822** | 2.395 | **2.934** |
| India | 883,188 (40.02%) | 809,986 (46.75%) | 2.285 | **2.855** | 2.395 | **2.969** |
| France | — | **259,452 (14.98%)** | — | 2.711 | — | 2.820 |

- The country mix shifts strongly toward India and France. Validation must be re-weighted (§4.7).
- **Pool-per-S1 grows by about 24% in every country.** Whether this comes from more distractors or
  from larger fan-out is `[UNSPECIFIED]` (test labels are hidden).
  - Under equal fan-out, the test distractor share ≈ 1 − (1.674 / 2.82) ≈ **41%** for S2, versus
    26.6% in train `[INFERENCE]`.
  - Both scenarios are simulated in validation (§4.7).
- Test S1 addresses are slightly longer (median 50 vs 41 characters). French addresses are long
  (`… Rue …, City, Region`).

### 6.4 Ground-truth structure `[DATA]`

- **Match-list length distribution** (share of train S1 entities):

  | Matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Share | 5.58% | 5.40% | 17.00% | 24.05% | 21.94% | 14.59% | 7.47% | 2.90% | 0.85% | 0.19% | 0.02% | 37 entities |

- Mean list length among non-singletons: **3.666**.
- **Singleton rate by country:** India 5.59%, US 5.58%. It is not country-dependent, which makes a
  similar test rate plausible `[INFERENCE]`.
- **Per source:**
  - an S1 entity has at least one S2 match 86.96% of the time and at least one S3 match 87.93%;
  - both sources 80.48%, S2 only 6.48%, S3 only 7.45%;
  - average matched S2 per S1 is 1.674, and S3 is 1.788.
- **Most common (#S2, #S3) patterns:** (1,1) 12.2%, (1,2) 11.4%, (2,1) 10.1%, (2,2) 9.4%,
  (1,3) 6.4%, (0,0) 5.6%.
- **Exclusivity:** 7,638,365 matched IDs, all distinct, so each S2/S3 record belongs to at most one
  S1.
- **Distractors:** 26.64% of train S2 and 25.37% of train S3 records match no S1 entity.
- **Country agreement on matched pairs: 100.000%** (S2: 3,693,619 of 3,693,619; S3: 3,944,746 of
  3,944,746).

### 6.5 Leakage audit `[DATA]`

| Check | Result | Conclusion |
|---|---|---|
| Spearman(ground-truth row order, S1 file order) | 0.0003 | No ordering artefact |
| Spearman(S1 file position, matched S2 / S3 position) | 0.0010 / −0.0007 | No positional leakage |
| Spearman(S1 numeric ID, matched S2 / S3 numeric ID) | 0.0001 / 0.0002 | IDs are random |
| Within-cluster row gap in S2 / S3 (median) | 1.09M / 1.11M rows (random expectation about 1.7M); share with gap ≤ 5 rows = 0.0000 | Records of one entity are not adjacent |
| Train/test ID overlap (S1 / S2 / S3) | 0 / 0 / 0 | Disjoint splits |
| Test S1 (name \| address) exactly present in train S1 | 0 | No duplicated entities |
| Test S1 lowercase *name* present in train S1 names | 585,588 of 1,732,544 (33.8%) | **Shared name-generator vocabulary** (not label leakage). It means transliteration dictionaries and appendix lists learned from train will transfer. |

**Residual leakage risks are self-inflicted ones** (listed in §7.1): fitting label-derived
artefacts on validation entities, non-OOF stacking, and tuning on the public LB.

### 6.6 Noise taxonomy: real examples from the sampled clusters `[DATA]`

Each line shows an S1 record → a matched S2/S3 record from the ground truth.

**Name noise**

| Pattern | Examples |
|---|---|
| Legal-suffix swap, drop or add | `Capital Holding, LLC` → `Capital LLC Services`; `Professional Mobility Global Inc.` → `…Global-Incorporated`; `Bachmann's Bistro` → `Bachmann's Bistro Ltd` |
| Generic appendix | `Krishna Finance Private Limited` → `Krishna Finance Limited Center`; `Supreme Lifeco, Inc` → `Supreme Inc Partners`; `Mahalakshmi …` → `Sri Mahalakshmi Private Limited Sérvice` |
| Word reordering | `Shree Entertainment Pvt Ltd` → `Shree Ltd Pvt Entertainment`; `Ulhashnagar Edge Ltd` → `Edge Ulhashnagar Ltd`; `Bpl Technologies LLP` → `L.L.P. Bpl Technologies` |
| Typos, including heavy corruption | `Capital Holdig`; `Wildlife (Cotmmite)`; `Kanva Seeovces`, `KANVA SEHVCCIES`; `MAHALAKSHMI ENTERPRISES PDRVIMTSE`; `U1hashnagar` (1↔l) |
| Injected accents | `Esparza Ridge Sunrise Ínc`, `Rídge`, `Éntertainment`, `Báchmann'S` |
| Honorific prefixes | `Mr Jagdamba …`, `M/s JAGDAMBA …`, `Sri Shree …`, `Shri Balaji …`, `Smt viraladvertising.com`, `The D and J Scientific` |
| Junk prefixes | `>> Dss Care L.L.P.`, `... Balaji Projects`, `-- Holloway Peak`, `<< Team Ecole`, `Infotech -- MD Private` |
| Website and handle forms | `capitalholding.com`, `PERLASSEAFOOD.COM`, `#wildlifecommittee`, `@bachmannsbistro`, `UrologypartnersCom`, `Gmustang.Com` (initial + word), `bstrategy.com` (`Bhopal Strategy`) |
| Acronyms | `Gallagher Crystal John LLC` → `GCJ`; `Emm College Private Limited` → `EC` |
| DBA with an invented alias in front | `Jaxevo trading as Mansfield Devices`, `Synveo a/k/a Jain Star …`, `Wexbelozeta f/k/a Supreme Lifeco, Inc`, `Ectolyra formerly Internal Medicine Grand Clinic Inc` |
| **Fully invented replacement name** (only the address links) | `Mccoy and Bryan` → `Brixcalo`; `Peralta Cardiology` → `Néxzephnex`; `Cascade College` → `Ariaaria`; `Mahalakshmi Enterprises …` → `Nexavi` |
| Native script, full or partial | `Krishna Finance Private Limited` → `कृष्ण फाइनेंस प्राइवेट लिमिटेड`; `City Producer …` → `સિટી પ્રોડ્યુસર …` (Gujarati); `Apex Consulting` → `অ্যাপেক্স কনসাল্টিং` (Bengali); `Blue Developers Limited` → `Blue Developers लिमिटेड` |
| Casing | `ULHASHNAGAR EDGE LTD.`, `foot & ankle cornerstone clinic inc` |
| `&` / `+` / `and` | `Ear Nose + Throat`, `Foot + Ankle`, `The D and J Scientific` |
| Brackets | `Wildlife [Committee]`, `Professional Mobility (Glóbal)`, `Jain Star Products Private (Limited)` |

**Address noise**

| Pattern | Examples |
|---|---|
| Abbreviation and casing | `Crystal Mist Drive` → `CRYSTAL MIST DR`; `Falling Leaf Lane` → `FALLING LEAF LN`; `Delaware Turnpike` → `Delaware Tpke` |
| Wrong expansion | `Hamilton Street` → `HAMILTON SAINT` |
| Component reordering | `OH, 108 2nd Street, Rome` vs `STOUT, OH, 108 SECOND ST`; `Texas, Houston, #9 Falling Leaf Lane` |
| State format | `TX` ↔ `Texas`; `Maharashtra` ↔ `MH` ↔ `महाराष्ट्र`; `Tamil Nadu` ↔ `TN` ↔ `தமிழ்நாடு`; `Gujarat` ↔ `GJ` ↔ `ગુજરાત`; `Bihar` ↔ `BR` ↔ `बिहार` |
| **City substitution** | Rome ↔ Stout; Fairview Park ↔ Cleveland; Rotterdam ↔ Schenectady; Memphis ↔ Cordova; `City Of Kiel` ↔ `KIEL`; `NEW FAIRFIELD CITY` |
| House-number edits | Zero padding `008644`, `0016`; ranges `2210-2214`, `108-110`; prefixes `#9`, `##7A`, `H.no 845`, `No. 398 44`, `D/123/3`, `C-64/9`, `G-226`; **digit drops** `3280` → `328`, `3488 85` → `488 85`, `2411` → `241`; separators `4-`; `343A` |
| Ordinals | `2nd` → `SECOND`, `2ST`, `2rd`; `8th` → `EGIHTH`; `4th` → `FOURTH` |
| Injected PO Box / PMB | `PMB 3330`, `PO BOX 4187` (different numbers on records of the same entity) |
| Null tokens | `NULL`, `null`, `N/A`, and an `INCORPORATED` token inside an address |
| Missing components | Number dropped (`Crystal Mist Dr, Tucson`), street dropped (`G-226, Mumbai, Mumbai City, महाराष्ट्र`), **entire address empty** |
| Typos | `SURVLY`, `Holse`, `Hosue`, `Hose`, `Decorha`, `OTTUMMWA`, `Centarlia`, `Eisenhoewr`, `LMIA` |
| Landmarks | `Near Changodar`, `Opp: Sakar Baug`, `Near Riddhi Siddhi`, `Near Fortis Hospital` |

**France (test only) `[DATA]`**

| Pattern | Examples |
|---|---|
| Legal forms | `SARL`, `SAS`, `SASU`, `EURL`, `SA`, `SCI`, `SNC`, dotted (`S.A.R.L.`, `S.A.S.`, `S.N.C.`) |
| Street types | `Rue`/`R.`/`R`, `Avenue`/`Av`, `Boulevard`/`BD.`, `Allée`/`ALL.`, `Impasse`, `Chemin`, `Cité`, `Corniche` |
| Number formats | `N° 50`, `NO 14`, `53 Bis`, `2 D Rue`, `17 G Rue`, `9 - R. DU TILLEUL`, `0034 R.` |
| Region ↔ département | `Nouvelle-Aquitaine` ↔ `Gironde`; `Hauts-de-France` ↔ `Nord`/`Pas-de-Calais`; `Pays de la Loire` ↔ `Loire-Atlantique` |
| Shared noise generator | Injected accents (`Immôbiliere`, `Fôyer`, `Spôrtive`), `(France)` insertion, handles (`fmusique.com`, `CORALIEFETESSASCOM`), acronyms (`TC`), typos (`Doctour`, `GAUCE`), escaped quotes (`"""centre Ateliers"`) |
| **Extreme generic-name reuse** | `Bordeaux Club SARL` ×205 test S1 entities at different addresses; `nantes club sarl` ×157; `lille club sarl` ×147 |

### 6.7 Generic names = structural hard negatives `[DATA]`

| | Train S1 | Test S1 |
|---|---|---|
| Distinct lowercase names | 1,538,804 of 2,206,821 rows | 1,238,244 of 1,732,544 rows |
| Names used by more than one entity | 178,080 | 130,383 |
| Most repeated | `primary care group` ×253, `ear nose & throat group` ×251 | `bordeaux club sarl` ×205 |

- S2/S3 have their own generic short names: `primary care` ×397–421, `urgent care`, `internal
  medicine`, and **two-letter names** (`cc` ×302–387, `pc`, `sc`, `ac`, `lc`, `mc`), which look
  like acronyms of generic names.
- **Consequence:** name-equality evidence must be discounted by name frequency (F-group features),
  and address evidence decides these cases.

### 6.8 Signal separation: matched vs random pairs `[DATA]`

Sample: 30,000 ground-truth clusters → 104,069 matched pairs, plus the same number of random
same-country, same-source pairs.

**Matched pairs**

| Country, source | Exact lowercase name | Normalised names equal | Share ≥ 1 core name token | Native-script name | Name token-set ratio p10 / p25 / p50 | Address token-set ratio p10 / p25 / p50 | Share a number (both have numbers) |
|---|---:|---:|---:|---:|---|---|---:|
| India S2 | 6.1% | 44.7% | 70.7% | 23.8% | 11 / 44 / 93 | **86 / 93 / 100** | 96.7% |
| India S3 | 6.5% | 46.7% | 80.2% | 12.9% | 13 / 82 / 97 | 74 / 88 / 94 | 96.0% |
| US S2 | 14.0% | 58.6% | 91.5% | 0% | 78 / 93 / 100 | 76 / 89 / 95 | 87.4% |
| US S3 | 12.8% | 55.1% | 91.8% | 0% | 79 / 93 / 100 | 70 / 81 / 88 | 90.0% |

**Random pairs:** they share a core name token about 1.2% of the time. Name token-set ratio p90 is
43–64, address token-set ratio p90 is 43–46, and they share a number 0.2–8.7% of the time.

**8,295 of all 104,069 matched pairs (8.0%)** have Latin script on both sides and still share
**no** core name token. These are handles, invented names and acronyms, and they are recoverable
only through the address.

**Takeaways:**

- (a) Against *random* pairs, both fields separate almost perfectly. The real difficulty lies in
  *hard* negatives: same generic name, nearby address.
- (b) The address is the more reliable single field.
- (c) For India, the name side needs transliteration.

### 6.9 Baseline blocking experiment `[DATA]`

**Setup.**

- 2,000 random train S1 queries per country, searched against the **full** same-country train
  S2 and S3 pools: US 3.02M / 3.17M, India 2.02M / 2.12M records.
- Char-3-gram TF-IDF with crude normalisation (`eda_block.py`): legal forms and honorifics
  dropped, DBA split, TLD strip, abbreviation expansion, PO Box removed. There is **no**
  transliteration and **no** learned state map.
- Recall is the share of true S2/S3 matches found within the top-K per S1, per source.
- The retrieval is verified against an independent exact brute force (§L3 lesson).

**Recall of true matches within the top-K per S1**

| Country, source | K | Name-only | Address-only | Union |
|---|---:|---:|---:|---:|
| US S2 | 5 / 10 / 20 / 50 | 58.6 / 65.8 / 70.8 / 76.8% | 83.7 / 88.9 / 91.3 / 92.1% | 93.5 / 96.5 / **97.7** / 98.7% |
| US S3 | 5 / 10 / 20 / 50 | 57.5 / 65.2 / 71.1 / 76.8% | 82.8 / 88.0 / 90.3 / 91.4% | 93.3 / 96.0 / **97.3** / 98.3% |
| India S2 | 5 / 10 / 20 / 50 | 46.8 / 52.1 / 57.5 / 63.0% | 83.0 / 86.6 / 88.4 / 89.9% | 89.6 / 92.2 / **93.7** / 95.1% |
| India S3 | 5 / 10 / 20 / 50 | 46.5 / 51.5 / 57.1 / 66.1% | 72.0 / 76.1 / 79.0 / 82.1% | 85.8 / 89.2 / **91.7** / 94.6% |

**Totals at name@20 ∪ address@20 (both sources)**

| Country | Pair recall | Candidates per S1 | Oracle F0.5 | Naive threshold F0.5 |
|---|---:|---:|---:|---:|
| US | **97.48%** | 77.1 | **0.9920** | 0.7477 (w_name = 0.4, t = 0.72) |
| India | **92.67%** | 78.0 | **0.9703** | 0.6839 (w_name = 0.3, t = 0.68) |
| Test-mix-weighted US + India (0.3827 : 0.4675, renormalised; France excluded) | — | — | **≈ 0.980** | ≈ 0.713 |

**Name-only diagnosis** (`eda_block_diag.py`, 300 US queries, exact brute force):

- Name-only recall@20 is 79.8% when ties at the K-th score count as hits.
- **Every miss had ≥ K other pool records scoring strictly higher than the true match.**
- The number of pool records sharing the query's *exact* normalised core name: median 2, p75 6,
  **p90 70**.
- Typical misses: `Fresh Auto Body LLC` → true `Fresh Auto` (cosine 0.79), out-ranked by ≥ 20
  other entities' `Fresh Auto Body LLC` records at cosine 1.0; `Seneex Holdings` → true `5eneex`
  (digit substitution), out-ranked by `Leneex Holdings` and `Neex Holdings`.

**What this means for the design:**

1. **The address is the stronger single retrieval key in both countries** (up to 91% vs 71% at
   K=20). Name-only retrieval *saturates* (it barely improves from K=20 to K=50) because
   generic names crowd the top-K with other entities' records.
2. **The union is far better than either pass.** They fail on different records (invented or
   handle names vs. empty or truncated addresses). That is the rationale for the multi-pass
   design in §L3.
3. **US blocking is essentially solved** even crudely: the oracle is 0.992, above the leaderboard
   top of 0.984. **India is not**: the oracle is 0.970. India name recall is held back by native
   script (about 24% of India-S2 matches; transliteration pass P4) and India-S3 address recall
   (79%) by state codes, native-script state names and truncated addresses (the learned state
   map).
4. **The naive threshold baseline is far below the oracle** (0.75 vs 0.99 in the US). Most of the
   gap must be closed by the learned matcher (L4/L5) and the decision layer (L6).
5. **Caveats:**
   - 2,000 queries per country, so recall standard errors are about ±0.3 pp.
   - The naive thresholds were tuned on the same queries (slightly optimistic).
   - The pool density is train-like (about 26% distractors), not the denser test (§6.3), which
     makes real test retrieval somewhat harder.
   - The candidates per S1 (about 77) count both sources and both passes before any pruning.

### 6.10 Baseline ladder

| Baseline | Macro F0.5 | Where measured |
|---|---|---|
| Predict all empty | **0.0558** | Train ground truth, exact |
| Naive TF-IDF threshold (no ML, no assignment) | **US 0.748, India 0.684** (≈ 0.713 test-mix, US + India) | §6.9 sample |
| Oracle given the naive name@20 ∪ address@20 candidates | **US 0.992, India 0.970** (≈ 0.980 test-mix, US + India) | §6.9 sample. This is the *ceiling* a perfect classifier could reach with those candidates. |

| *External reference:* public LB top 3 on day 1 | **0.984098 / 0.98377 / 0.980473** | `[LB]`: a public-subset test score (includes France), about 14 h into the challenge |

Every model must be reported against this ladder, per country and test-mix weighted.

**What the leaderboard implies** `[LB]` + `[INFERENCE]`:

- (1) The theoretical maximum is 1.0. Top teams already sit about 1.6 points below it, so the
  irreducible error (for example, an invented name with an empty address, or a generic name at a
  shared address) is at most about 1.6% of the entity-level F.
- (2) France is evidently solvable at high accuracy by a well-built pipeline, despite being
  zero-shot.
- (3) With naive blocking, our measured oracle ceiling (§6.9) is **0.992 for the US but only
  0.970 for India**, which is about 0.980 test-mix-weighted, *at or below* the leader. So **both
  gaps matter**:
  - the matcher and decision layer (naive 0.71–0.75 → oracle);
  - **India blocking**: transliteration, name+locality and reverse passes are needed to lift the
    ceiling above about 0.985.
- (4) The public LB is a subset and the final ranking is private (§1.14). Differences of
  about 0.001 at the top are within split noise.

### 6.11 Not yet measured: executable audit checklist `[STRATEGY]`

| # | Item | Why it matters | How |
|---|---|---|---|
| 1 | Address sharing across distinct S1 entities (exact and normalised) | Size of the "same building, different business" hard-negative class; the weight of address-only evidence | Group S1 by normalised address; count groups > 1 |
| 2 | S2↔S3 sibling similarity within clusters vs across | Value of graph consistency (L6-5) | Pairwise features within GT clusters vs nearest non-cluster records |
| 3 | Transliteration dictionary coverage on **test** native tokens | Expected India-name recall | Learn on train; OOV rate on test S2/S3 native tokens |
| 4 | France profile (legal-form mix, share of number formats, region/département mix, generic-name share) | Zero-shot rule pack | Same profiler, restricted to `country == France` |
| 5 | Blocking recall per pass after proper normalisation (P1–P7) | Recall ceiling | Extend `eda_block.py` |
| 6 | Candidate-rank distribution of true matches after the pre-ranker | Choice of the budget N | Oracle-F-vs-N curve |
| 7 | How often a true match's number set **conflicts** with the S1 number set | Whether "number conflict" may be near-hard | Pair features on GT pairs |
| 8 | Accuracy of the city-substitution pattern | Whether city mismatch is soft evidence only | Share of GT pairs with a city mismatch |
| 9 | Stability of distributions between public and private LB subsets | Unknowable (`[UNSPECIFIED]`) | Monitor the LB-vs-validation offset |

### 6.12 Conditional preprocessing decisions `[STRATEGY]`

| If the audit shows… | …then |
|---|---|
| Item 1: many S1 entities share an address | Require name evidence for address-only links; add an address-frequency feature (already planned); lower the weight of pass P2 in the pre-ranker |
| Item 3: dictionary OOV > 20% on test | Invest in the character model and the phonetic key; rely more on pass P6 (dense multilingual) |
| Item 4: French legal forms appear at a very different rate than US/India forms | Keep legal form out of the matcher's features (so it is not trusted zero-shot) or map it to generic categories (present / absent / compatible) |
| Item 5: P6 adds < 0.5 pp recall over P1–P5 | Drop dense blocking for cost; keep embeddings only as features |
| Item 7: number conflicts in GT pairs < 1% | Use number conflict as a near-veto (it is a strong precision gain); otherwise keep it soft |
| Item 8: city mismatch in ≥ 5% of GT pairs | Never filter on city (already the plan); learn city co-occurrence aliases |
| Empty-address share on test is materially higher than on train | Re-weight name features and the singleton prior for those records |

---

## PART 7 — ADDITIONAL CONSIDERATIONS, RISK REGISTER AND FINAL AUDIT

### 7.1 Leakage risks specific to this ER task

| # | Risk | Mechanism | Guard |
|---|---|---|---|
| LK1 | Label-derived artefacts fitted on validation entities | The transliteration dictionary, appendix-token list, state map, pre-ranker or cross-encoder learned from ground-truth pairs that include validation clusters inflates validation F | Learn **inside each fold** from training-fold clusters only. Refit on all train for the final test run. |
| LK2 | Non-OOF stacking | A second-stage model trained on in-sample first-stage predictions | Strict out-of-fold predictions for every stacked input |
| LK3 | TF-IDF/IDF or frequency statistics fitted on a different pool than the one scored | Inconsistent feature distributions between validation and test | Fit unsupervised statistics on the pool being scored (validation pool for validation, test pool for test). Label-free and documented as transductive (§1.13). |
| LK4 | ID or row-order artefacts | A model latches onto ID digits or positions | Verified uninformative (§6.5), and **excluded** from features anyway |
| LK5 | Threshold tuning on the public LB | Overfits a subset; the final ranking is private | Thresholds come from OOF only. The LB is a monitor, not an optimiser. |
| LK6 | Validation pool easier than test | 26% vs about 41% distractors | Dense-distractor validation variant (§4.7) |
| LK7 | Train/test entity overlap | Would reward memorisation | None found (0 ID overlap, 0 exact name\|address overlap) `[DATA]` |
| LK8 | Assuming the test singleton rate or fan-out equals train | The decision layer tuned to wrong priors | Robustness sweep across priors; the expected-F selection uses the candidates' own evidence, not a fixed prior |

### 7.2 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Output rejected (missing S1 row, CRLF, bad ID) | Medium without guards | Lost submission | L8 writer iterates the full S1 list; LF-only; byte and ID assertions; validator `--check-ids` |
| Blocking recall ceiling too low | **Measured:** name-only@20 recalls only 57–71% of pairs; even name ∪ address@20 caps India at an oracle of 0.970, below the LB top of 0.984 (§6.9) | Caps every downstream gain, especially India (47% of test) | Transliteration pass (P4), name+locality (P3), reverse (P7), dense (P6), learned state map; measure the marginal gain of every pass per country |
| France underperforms | High (zero-shot) | 15% of the test score | Country-agnostic features; LOCO-driven selection; French rule pack; transductive IDF; pseudo-labelling validated by LOCO simulation |
| Denser test distractors → false merges | Medium–high | Precision loss across all countries | Exclusive assignment; competition features; dense-distractor validation |
| Memory exhaustion (16 GB laptop; 3 GB free during this analysis) | High at full scale | Crashes, lost time | Per-partition processing; pyarrow; float32; offload to a memory-optimised Processing job |
| **Silently wrong retrieval or scoring code** | **Observed during this analysis** (CSC `.indices` misread as columns → recall under-measured about 4×; a non-independent self-check passed anyway) | Wrong design decisions; hidden recall loss | Assert fast top-K against an *independent*, unrestricted brute force on sampled queries in every run; enable library invariant checks; unit-test the scorer against the PS example |
| GPU unavailable locally (CPU-only torch) | Certain until fixed | Neural components too slow | Install a CUDA build, or use a SageMaker GPU job |
| Licence violation in a pretrained component | Low if the register is kept | Disqualification at review | Licence register (Phase 0); only MIT/Apache models; total ≤ 8B |
| Fair-play perception (grey areas) | Low–medium | Review friction | No network calls at runtime except the pinned model download; every grey-area choice documented |
| Credits exhausted or idle billing | Medium | Lost compute | Budgets alarms; auto-stopping jobs; no endpoints; stop notebooks |
| Team coordination (only the leader submits; no simultaneous logins) | Medium | Missed submission slots | The leader owns L8 and the ledger; a pre-agreed submission schedule (§3.3) |
| Non-reproducible final zip | Medium | Disqualification or penalty at review | Clean-environment dry run; pinned requirements; configs; seeds; hashes |
| Overfitting to the public LB | Medium | Private drop | §3.3 policy; robust configurations favoured |

### 7.3 Final completeness self-audit (master prompt §10)

| Question | Answer | Justification / location |
|---|---|---|
| Every sentence of the PS addressed, including France zero-shot, the matches ⊆ candidates rule and the MIT/Apache ≤8B constraint? | **Yes** | §1.1–§1.13; France: §1.3, §4.8/L7-5, §6.3/§6.6; subset rule: §1.7, §1.17 C4, L3, L8; licence: §1.10, the L5 licence register |
| Every official guideline addressed, including 5 submissions/day and team-leader-only submission? | **Yes** | §1.14, §3.3; team leader `[UNSTOP]`: §1.16, §2.3, §7.2 |
| Both deliverable files, the zip structure and the documentation template covered? | **Yes** | §1.7, §1.9, L8, L9 (template mapping), Phase 12 |
| Flagged that the dataset was (not) provided, instead of fabricating statistics? | **Yes, updated:** the data *was* provided and was measured; the plan-only items are listed | §0.2, Part 6 status flag, §6.11 |
| ER-specific leakage risks addressed (TF-IDF on test vocabulary, ground-truth-derived features, ID ordering)? | **Yes** | §7.1 LK1–LK8; §6.5; L4 leakage notes |
| Baseline established and tied to macro F0.5? | **Yes** | §6.10 ladder: all-empty 0.0558 (exact), the naive threshold baseline and the oracle, measured |
| Model choices justified by the data (transliteration, three-country address formats, noisy short text)? | **Yes** | L2 (native-script shares, learned transliteration), L3 (address blocking from §6.8/§6.9), L5 (GBDT rationale, multilingual encoders) |
| Precision-heavy F0.5 handled in thresholding and decisions? | **Yes** | §1.11 count form and rank-dependent break-evens; L6 (assignment, rank thresholds, expected-F selection) |
| Explained why no hosted endpoint is needed, citing the workshop and the AWS-FAQ? | **Yes** | §5.1, §5.2, §5.3 item 11 |
| External lookup ban treated as an architectural constraint? | **Yes** | §1.13 (and its grey areas); L2 (everything learned from the provided files); §5.2 (Bedrock excluded) |
| Facts separated from strategy and inference throughout? | **Yes** | Tags on every claim; the added `[DATA]` tag; unverifiable `[UNSTOP]`/`[WORKSHOP]` claims flagged in §0.3 and §1.16 |
| Avoided shrinking the solution to fit 72 hours? | **Yes** | The full L1–L9 design and all phases are described first; tiering (§3.2) and the live-window ordering note (§3.3) are explicitly *after* it |

---

## APPENDIX A — REPRODUCING EVERY `[DATA]` NUMBER

All scripts are in `amazon_ml_2026_analysis/eda/`, and their raw outputs are next to them
(`*_out.txt`).

- **Environment:** Windows 11, Python 3.12.3, pyarrow 23.0.1, numpy 2.4.2, scikit-learn 1.8.0,
  RapidFuzz 3.14.6, torch 2.11.0+cpu.
- **Hardware:** 12 logical CPUs, 16 GB RAM, RTX 3050 6 GB (unused by the CPU-only torch build).
- **Data location:** each script reads the dataset from the hard-coded `BASE` path
  `C:\Users\ANEXUS\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset`.

| Script | Produces | Runtime (local) |
|---|---|---|
| `eda_profile.py [files…]` | §6.1–§6.2 per-file profiles, token and name frequencies | About 2–4 min per 5M-row file |
| `eda_gt.py` | §6.4 ground-truth structure, §6.5 leakage and overlap checks | About 5 min |
| `eda_pairs.py` | §6.6 sampled clusters, §6.8 pair signals (seed 0, 30,000 clusters) | About 5 min |
| `eda_block.py 2000` | §6.9 blocking recall, naive baseline, oracle (seed 0, 2,000 queries per country; includes an independent brute-force self-check) | About 25 min |
| `eda_block_diag.py` | §6.9 diagnosis of name-only recall (seed 1, 300 US queries vs full US S2) | About 5 min |

Shell-level checks (Git Bash) used for §6.1:

```bash
wc -l dataset/train/*.tsv dataset/test/*.tsv                    # row counts (+1 header)
grep -c '"' <file>                                              # lines with double quotes
awk -F'\t' 'NF!=4' <file> | wc -l                               # malformed rows (0 everywhere)
grep -c $'\r' <file>                                            # CRLF lines (= all lines)
```

## APPENDIX B — GLOSSARY

| Term | Meaning |
|---|---|
| Blocking / candidate generation | Cheap retrieval that limits which (S1, S2/S3) pairs the matcher scores |
| Pair completeness (PC) | Share of true pairs that survive blocking (the recall ceiling) |
| Reduction ratio (RR) | 1 − candidate pairs / all possible pairs |
| Oracle F0.5 | The macro F0.5 a perfect classifier would reach on the given candidates |
| Distractor | An S2/S3 record that matches no S1 entity |
| Exclusive assignment | Each S2/S3 record is linked to at most one S1 (a structural fact of the ground truth) |
| Expected-F0.5 set selection | Per S1, choose the candidate subset that maximises the expected F0.5 given calibrated probabilities |
| LOCO | Leave-one-country-out validation, used as a proxy for the unseen France |
| OOF | Out-of-fold predictions, the only legitimate inputs for stacking or calibration |
| Transductive | Uses the *unlabeled* inputs being scored (for example, IDF on the test pool) but never test labels |
