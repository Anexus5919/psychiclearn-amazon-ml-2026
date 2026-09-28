# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** PsychicLearn  
**Team Members:** Adarsh Singh (team leader), Atharva Waghmode, Sanjog Poojary, Atharva Gadge  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Our pipeline has four parts:
- **Hybrid candidate generation.** Lexical TF-IDF search, exact-name search and state-restricted
  search are combined with a **fine-tuned multilingual dense bi-encoder**. Together they reach 99.55%
  pair recall.
- **Learned pruning** down to ~8 candidates per business.
- **A LightGBM matcher** that stacks ~110 hand-built similarity features with the probabilities of
  **two fine-tuned cross-encoders**.
- **Exclusive assignment** with F0.5-tuned thresholds.

Our best submission scores **0.982608** on the public leaderboard, with **0.98769** macro F0.5 on
304,555 held-out training businesses. Every learned component is trained only on the provided data,
with the validation businesses held out.

---

## 2. Methodology

### 2.1 Problem Analysis

We profiled every row of every file before modelling.

**Structure:**
- 5.6% of S1 businesses have no match, and matched businesses average 3.67 matches.
- Each S2/S3 record belongs to **at most one** S1 business.
- Matched pairs **always share the country**.
- The test set is 47% India, 38% US and **15% France, which has no training data** (zero-shot). It
  has about 24% more distractor records per business than training.

**Noise types found:**
- Indic scripts: about 24% of India S2 names are in Devanagari, Bengali, Gujarati, Kannada and
  others, often with the address cut to "door number, city, state code".
- Look-alike digits (`Internati0na1`), injected tags (`(ID: 64721)`, `***`, `Mr`), and appended
  generic words (Service, Center, Partners).
- Dropped or duplicated words (`Pvt Pvt Ltd`), legal-form variants (Pvt/Private, SARL, S.A.S.).
- Abbreviated, reordered or truncated addresses; state codes (MH, KA) and French départements
  instead of regions.
- Empty addresses (3–4% of true matches, 19–46% of the matches our run-4 retrieval missed).
- Names replaced by gibberish at the same address.

**The key difficulty is crowding.** Generic names repeat hundreds of times per country:
- 24% of India test records and 19.5% of France test records (Source 2) share their exact name with
  more than 15 others.
- In run 4, 76% of India's missed true matches had near-identical names but were pushed out of the
  top-k by same-named businesses elsewhere.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking (lexical + neural dense retrieval), then a learned pruner, then a
gradient-boosted classifier stacked with neural cross-encoders, then a constrained assignment.

**Core Innovation:**
- **Fine-tuned dense retrieval against crowding and script changes.** It raised India's retrieval
  recall from 0.947 to 0.997 and France's inferred score by about 3.6 points, zero-shot.
- **Leak-free stacking.** Both cross-encoders are trained on businesses that the LightGBM never sees.
- **Everything learned from the training pairs:** the transliteration dictionary and the state/region
  map.

---

## 3. Candidate Generation (Blocking)

The country is a hard block. The candidates are the union of these passes, per country and target
source:

- **Blocking keys used:**
  1. **TF-IDF name** (character 3-grams, top-10; top-15 for India);
  2. **TF-IDF address** (word tokens, top-10; top-15 for India);
  3. **TF-IDF name+address** (top-15; top-20 for India);
  4. **exact normalised core name** (names shared by at most 50 records);
  5. **name search restricted to the same state/region**, learned from S1 addresses and training
     pairs (98.4% (India) / 93.1% (US) region agreement on held-out true pairs);
  6. **dense bi-encoder**: `multilingual-e5-small` fine-tuned contrastively on 2M training pairs,
     15 nearest neighbours by cosine.

  Before retrieval, Indic scripts are transliterated with a learned dictionary, and look-alike digits
  and `ID` tags are removed from names.
- **Candidate pairs generated:**
  - test retrieval produced 148.7M pairs;
  - **learned pruning** kept 15.7M (8.3 per business);
  - `candidate_pairs.tsv` lists 11.0M (6.35 per business), keeping pairs with final matcher p ≥ 1e-4.
- **How you ensured true matches were not lost:**
  - Validation **pair recall of retrieval is 0.9955**. The ceiling for a perfect matcher on our
    candidates is 0.9987 (India) and 0.9987 (US).
  - The pruner keeps the smallest rule that loses **≤0.2%** of retrieved true pairs.
  - The final trim to p ≥ 1e-4 lost 0.003% of true pairs (measured on run-4 validation).
  - Every fast top-k search is checked against an independent brute force on sampled rows.

---

## 4. Matching Model

**Features used (~110):**
- **Name features:**
  - RapidFuzz ratio, token-set, token-sort, partial ratio, Jaro-Winkler;
  - no-space and acronym matches;
  - legal-form equality and conflict;
  - duplicated-word and core-name equality checks;
  - name frequency (a generic-name penalty);
  - Indic-script and handle flags.
- **Address features:**
  - token-set, sort, partial and ratio similarity;
  - house-number Jaccard, primary-number equality and edit distance, number conflict and subset;
  - empty-address flags;
  - **region match** (+1 same state, −1 different, 0 unknown).
- **Other:**
  - retrieval cosines and per-pass ranks;
  - context within each business's candidate list (rank, gap to best, list size);
  - pre-ranker probability and group statistics;
  - dense rank and cosine;
  - **cross-encoder probabilities**, with rank, gap and count within the list, from a fine-tuned
    `multilingual-e5-small` (1.02M pairs) and a fine-tuned `mdeberta-v3-base`. Both are trained on
    3% of businesses kept apart from the matcher's training set.

**Model type:** LightGBM (binary), 4-fold cross-validation grouped by S1 business, early stopping on
each held-out fold. The cross-encoders are transformers (MIT-licensed, 118M and 280M parameters).

**Threshold selection method:**
- Out-of-fold probabilities feed an **exclusive assignment**: each S2/S3 record goes only to the S1
  that scores it highest.
- **Rank-dependent thresholds** (first match t1 = 0.62, further matches t2 = 0.78) are grid-searched
  to maximise out-of-fold **macro F0.5**, counting matches that are missing from the candidates as
  false negatives.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.98769** validation (India 0.98772, US 0.98767, out-of-fold on 304,555
  businesses); **0.982608** public leaderboard. France is inferred at about 0.954 from the leaderboard.

  | Run | Main change | Validation | Public LB |
  |---|---|---|---|
  | 2 | lexical retrieval + LightGBM + exclusive assignment | 0.9515 | 0.931 |
  | 3 | + learned transliteration, learned pruning | 0.9613 | 0.941 |
  | 4 | + e5 cross-encoder, larger India search, more training data | 0.9763 | 0.963 |
  | 5 | + region-restricted search, exact-name pass, name clean-up | 0.9779 | 0.967 |
  | 6 | + fine-tuned dense retrieval, + mDeBERTa cross-encoder | **0.9877** | **0.9826** |

- **Common false positives (wrong merges):**
  - look-alike businesses at the same address (gibberish or different names at a shared building);
  - near-duplicate names ("Consulting" vs "Consultancy");
  - the same generic name in the same city;
  - legal-form conflicts (Pvt Ltd vs Limited) on otherwise identical records.

  These are confident errors (in our run-3 analysis only 0.2% of candidates had p between 0.65 and
  0.75), so threshold tuning can't fix them.
- **Common false negatives (missed matches):**
  - candidates with an empty address and a heavily altered name (a word dropped plus a generic word
    appended);
  - names replaced by gibberish that are linked only by the address;
  - later matches of large clusters rejected by the stricter t2.

  In run 4, before dense retrieval, the dominant loss was **retrieval** (5.6% of India's true matches
  never retrieved). In run 6, retrieval misses fell to 0.45%, and most of the remaining loss is the
  matcher's.

---

## 6. Conclusion

- **Blocking recall** was the main bottleneck: generic names crowd out true matches, and Indic
  scripts defeat lexical keys.
- **The biggest single gains came from neural components trained on the provided pairs.**
  - A fine-tuned dense retriever, stacked with cross-encoders into a GBDT, took us from 0.941 to
    0.9826 on the leaderboard.
  - It even worked zero-shot on France.
- **Lessons:**
  - Measure the recall ceiling before tuning decisions.
  - Keep every learned component leak-free.
  - Test ideas offline before spending uploads. France pseudo-labelling, for example, lowered a proxy
    score and was dropped.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`: all source is in `src/`, with `README.md` (exact reproduction
steps) and `requirements.txt` (pinned versions).

- **Entry point:** `python src/run_all.py --data-dir <dataset> --work-root <scratch> --out-dir <output>`.
  It runs, resumably:
  1. normalisation;
  2. base retrieval and pruning;
  3. cross-encoder data and training;
  4. final business selection;
  5. dense bi-encoder training and neighbours;
  6. region keys;
  7. retrieval;
  8. pruning;
  9. cross-encoder scoring;
  10. mDeBERTa training and scoring;
  11. feature augmentation;
  12. LightGBM training;
  13. prediction, writing and validation.

  It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- **Package `src/ber/`:**
  - `pipeline.py`: stages;
  - `normalize.py`, `translit.py`, `regions.py`;
  - `blocking.py`;
  - `prune.py`;
  - `features.py`, `features_extra.py`;
  - `cross_encoder.py`, `ce_data.py`, `ce_export.py`;
  - `queries.py`;
  - `model.py`, `decide.py`, `scoring.py`, `writer.py`.
- **GPU scripts `src/gpu/`:**
  - `dense_retrieval.py`;
  - `mdeberta_cross_encoder.py`, which also runs as a Kaggle notebook.
- **Tests:** `src/tests/`.
- A `--smoke` mode runs every step on a small dataset in about 20 minutes (checked on this package).
- **Compute:** a 16 GB laptop (12 threads, RTX 3050 6 GB) and free Kaggle GPUs (T4 ×2). No paid cloud.

### B. Additional Results

- **Retrieval ceiling** (a perfect matcher on our candidates), validation:
  - run 4: India 0.9770, US 0.9941;
  - run 5: India 0.9779, US 0.9953;
  - run 6: India **0.9987**, US **0.9987**.
- **Dense retrieval gate** (validation businesses, never seen in fine-tuning): of the true matches that
  run-5 retrieval missed, dense top-15 found **94.6%** (India) and **82.3%** (US).
- **Rejected ideas, measured offline:**
  - France pseudo-labelling: a proxy with the US as the "unlabelled" country gave −0.0012 F0.5;
  - per-source thresholds and expected-F0.5 set selection: +0.00004;
  - France-only threshold changes: under 2% change in French matches.
