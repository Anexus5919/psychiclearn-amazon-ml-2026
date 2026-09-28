# PsychicLearn: What We Did, Start to Finish (so far)

Amazon ML Challenge 2026, Business Entity Resolution. Written 26 Sep 2026, around 03:00 IST.
**Updated 26 Sep 2026, ~17:00 IST:** sections 9–14 are new (run 4, error analysis, team results, experiments, run 5).

| Upload | Time | What it was | Public leaderboard (F0.5) |
|---|---|---|---|
| #1 | 25 Sep, 23:59 | Run 2: first complete pipeline | **0.931** |
| #2 | 26 Sep, 02:50 | Run 3: + transliteration for India, + candidate pruning | **0.941** |
| #3 | 26 Sep, 15:44 | Run 4: + e5 cross-encoder, bigger India search, more training data, new features | **0.963** |

The leaderboard top was about 0.988 at the time of writing. Time remaining: about 45 hours (the deadline is 27 Sep, 23:59 IST).
*Update 26 Sep ~17:00:* leaderboard top 0.9906; about 31 hours remain.

---

## 1. The problem in simple words

We get business records from **three sources**:

- **Source 1 (S1)** is the clean reference list. Every business appears in it once.
- **Source 2 and Source 3 (S2, S3)** are messy copies from other "vendors". The same business can appear 0, 1 or several times there, with typos, abbreviations, reordered words, Hindi/Gujarati/Bengali script, missing addresses and so on.

**Task:** for every S1 record, list all the S2/S3 records that describe the same business. This is called **Entity Resolution (ER)**.

**How we are scored: macro F0.5.**

- For each S1 record we compute F0.5 = 1.25·P·R / (0.25·P + R), where P = precision and R = recall. Then we average over all S1 records.
- One wrong match (false positive) hurts as much as missing four correct ones. The metric is "precision-heavy".
- **Singletons** (S1 records with no match) score 1.0 only if we output an empty list.
- A perfect 1.0 needs every list exactly right: nothing missing and nothing extra.

**Data size:**

| | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2.2M | 5.0M | 5.3M |
| Test | 1.73M | 4.9M | 5.1M |

About 2.5 GB in total.

**Tricky part:** training covers only **US and India**. The test set adds **France**, a country the model never sees during training (zero-shot).

---

## 2. First, we understood the data (evidence before code)

We read all the official documents (problem statement, guidelines, AWS FAQ, the challenge video) and scanned **every row** of every file. Everything went into a full analysis document, `ER_Solution_Blueprint.md`. The findings that shaped the solution:

| Finding | Number | Why it mattered |
|---|---|---|
| S1 records with **no** match | only 5.6% | Predicting "no match" is almost always wrong. "Predict all empty" scores just 0.056. |
| Average matches per S1 record | 3.7 | We must find *several* matches per business, not just the best one |
| Each S2/S3 record belongs to at most **one** S1 record | 100% of 7.6M matches | A free precision trick: never give one record to two businesses |
| Matched pairs always have the same country | 100% | Country can split the problem into smaller independent parts |
| Test country mix | India 47%, US 38%, France 15% | Training was 60% US / 40% India, so the validation score needs re-weighting |
| Postal codes present | < 7% of addresses | Postal-code matching can't be used as the main key |
| India S2 names written in Indic scripts (Devanagari etc.) | ~24% | Names like "कृष्ण फाइनेंस" must be matched to "Krishna Finance" |
| Same generic name for different businesses | e.g. "Bordeaux Club SARL" ×205 | Name alone is not enough; the address is needed too |
| Records with an **invented** name (only the address links them) | ~8% of matches | The address is often the strongest clue |

We also caught a data trap: the files use Windows line endings (CRLF). If our output used them too, the last ID on every line would silently break. **We always write LF-only files.**

---

## 3. Compute: why the laptop and not AWS

- We set up the AWS CLI and signed in with `aws login` (no passwords or keys stored), then installed the AWS Agent Toolkit. We uploaded the data to a private S3 bucket (compressed to 1.04 GB).
- We then found that **AWS Free-plan accounts can only start tiny machines**: at most 2 CPUs and 8 GB RAM. The SageMaker quotas were 0. We requested increases, but the restriction is about the *plan*, not the quota.
- The only way around it is upgrading to the Paid plan, which **could charge real money**. Our rule is **no real money**, so we didn't. Also: credits can't be merged across teammates' accounts, and a Free-plan account **closes** when its credits hit $0.
- **Decision:** run the heavy work on the laptop (12 threads, 16 GB RAM). **AWS money spent so far: $0** (the $100 credits are untouched).

---

## 4. How our pipeline works (the approach)

Everything is written in Python. Runs 1–3 used **no external data, APIs or pretrained models**. *Correction (run 4 onwards):* we also use one **pretrained model**, `multilingual-e5-small` (MIT licence, 118M parameters, within the rules: MIT/Apache, ≤ 8B parameters). It is fine-tuned on the provided training data only (§9). Still no external data or APIs. The pipeline has 6 stages:

### Stage 1: Normalisation (cleaning the text)

Turn messy text into comparable text:

- **Names:**
  - lower-casing;
  - removing accents (only for Latin letters; Hindi vowel signs are kept);
  - removing junk like `>>`, `--`, `@`, `#`;
  - "X trading as Y" → Y;
  - `capitalholding.com` → `capitalholding`;
  - separating legal words (Pvt, Ltd, LLC, SARL, S.A.S…) from the core name.
- **Addresses:**
  - `Rd`/`Road` → `rd`, `St`/`Street` → `st`, French `R.` → `rue`;
  - `008644` → `8644`;
  - `2nd` → `2`;
  - removing PO Box / PMB numbers and `null` / `N/A` tokens.

### Stage 2: Blocking / candidate generation (finding possible matches quickly)

- Comparing 1.7M × 10M records directly is impossible, so for each S1 record we **retrieve the most similar S2/S3 records** in the same country.
- We use **TF-IDF** vectors plus **top-k search** (`sparse_dot_topn`), with three searches:
  - name: character 3-grams, which tolerate typos;
  - address: whole words;
  - name + address combined.
- Every search result is checked against an independent brute-force calculation on sample rows, to catch silent bugs.

### Stage 3: Features (describing each candidate pair with numbers)

About 50 numbers per pair:

- name similarity (RapidFuzz ratios, Jaro–Winkler);
- address similarity;
- house-number agreement or conflict;
- how common the name is (a generic-name penalty);
- search scores and ranks;
- context, e.g. the gap to the best candidate.

**Country is deliberately not a feature,** so that the model works on France.

### Stage 4: Model (LightGBM)

- A gradient-boosted tree model (LightGBM, MIT licence) predicts "same business?" for each pair.
- It is trained with **4-fold grouped cross-validation**: every training business is scored by a model that never saw it. That gives an honest validation score.

### Stage 5: Decisions (turning probabilities into lists)

- **Exclusive assignment:** each S2/S3 record goes only to the S1 record that scores it highest.
- **Rank-dependent thresholds:** the first match of a business needs less confidence than later matches. The maths of F0.5 says the first match is cheap to accept (because 94% of businesses have one) and extra matches must be surer. Both thresholds are tuned directly on the F0.5 score.

### Stage 6: Writing and checking the files

- Both TSVs are written with LF line endings, then checked against every rule.
- Finally the organisers' `validate_submission.py` is run with the strict ID check.

---

## 5. The runs, in order

### Run 1: too slow, stopped

- The first full run used character 3-grams for addresses too.
- After 47 minutes it had finished only a small piece. Projected total: **10+ hours**.
- **Fix:** a speed/accuracy benchmark showed that matching addresses by **whole words** was both **more accurate** (recall 0.873 vs 0.849) and **12× faster**.

### Run 2: first complete pipeline → **LB 0.931**

| Setting | Value |
|---|---|
| Candidates per S1 record | ~54 |
| Training examples | 8% of training businesses → 9.6M pairs, 584k true matches |
| Validation F0.5 | **0.9515** (US 0.973, India 0.920) |

- **Speed fix:** scoring 93M test pairs with 4 averaged models would have taken hours, so we switched to **one** fold model, which is 4× faster and consistent with how the thresholds were tuned.
- **What the leaderboard told us:** 0.931 with US 0.973 and India 0.920 implies **France ≈ 0.86–0.90**, our weakest country.

### Organisers' update (during the challenge)

> "candidate_pairs.tsv is part of your final submission… the approach that generates a smaller candidate set per Source 1 entity will be ranked higher."

So candidate-set size now matters for the final ranking, not only the score.

### Run 3: transliteration + pruning → **LB 0.941**

1. **Transliteration (India).**
   - We **learned** an Indic-script → English dictionary from the training data itself: about 507,000 aligned name pairs, e.g. `प्राइवेट → private`, `लिमिटेड → limited`, `महाराष्ट्र → maharashtra`.
   - A generic fallback converts unknown words letter by letter.
   - To keep validation honest, the dictionary was learned **only** from businesses not used for training or validation.
2. **Learned pruning (smaller candidate sets).**
   - A small, fast "pre-ranker" model scores the ~54 candidates using only cheap signals and keeps each business's best ones.
   - The keep rule is the smallest one that loses at most 0.2% of true matches.
   - Result: **~18 candidates per business instead of ~54**. The candidate file shrank from 1.2 GB to 424 MB.
   - The pre-ranker's score also became an extra feature for the main model.
3. **Results:**

| | Run 2 | Run 3 |
|---|---|---|
| Validation F0.5 | 0.9515 | **0.9613** |
| India | 0.9196 | **0.9443** (+2.5 points) |
| US | 0.9727 | 0.9726 |
| Candidates per business | 54 | **18** |
| Public leaderboard | 0.931 | **0.941** (we predicted ≈0.943) |

Our offline validation predicts the leaderboard well, so we can test ideas offline without wasting uploads.

**Size-vs-recall options found by the pre-ranker:**

| Allowed loss of true matches | Candidates per business |
|---|---|
| 0.2% (used) | 17.7 |
| 0.5% | 11.9 |
| 1.0% | 10.0 |

---

## 6. Problems we hit, and what we learned

| Problem | Fix / lesson |
|---|---|
| An analysis script measured name-search recall as 18% instead of the true ~80% (sparse-matrix format mix-up) | Always verify fast search code against an independent brute-force check |
| Address search with character 3-grams took 10+ hours | Benchmark before long runs; word-level search was better *and* faster |
| Test scoring with 4 models over 93M pairs took hours | Use 1 model; prune candidates so far fewer pairs are scored |
| An automatic "start the next stage" watcher waited on itself | Never detect a process by text that also appears in the watcher's own command |
| AWS Free plan only allows tiny machines | Laptop compute; AWS used only for storage; $0 spent |

---

## 7. Where everything is

| What | Where |
|---|---|
| Private GitHub repo | https://github.com/Anexus5919/psychiclearn-amazon-ml-2026 |
| Final-package folder (zip structure) | `Downloads\PsychicLearn_submission\` |
| Newest code (run 3) | `Downloads\PsychicLearn_dev\src\` |
| Run-3 output files (uploaded, LB 0.941) | `Downloads\PsychicLearn_run3_output\` |
| Full analysis and plan | `Downloads\amazon_ml_2026_analysis\ER_Solution_Blueprint.md` |
| Short handoff note | `Downloads\PsychicLearn_HANDOFF.md` |

---

## 8. What's next (as planned at 03:00; superseded by §13–§14)

| Next step | Why | Time (coding + compute) |
|---|---|---|
| **Fix France** | Weakest country (~0.86–0.90) and 15% of the test set | ~1.5 h |
| **Train on more data** (8% → ~25%) | Sharper decisions on look-alike businesses | ~2 h |
| **Group-level features** (second-stage model using sibling records' scores) | Records of one business look alike, so decide them together | ~1.5–2 h |
| Possibly a smaller candidate budget (0.5% loss → ~12 per business) | Candidate size counts in the final ranking | ~30 min |
| Final package: zip + filled documentation + final upload | Required deliverable | ~1–1.5 h |

Realistic target: **0.96–0.975**. Getting to ~0.985 depends on how much of France and the group structure we can fix in the time left.

---

## 9. Run 4: first neural model (e5 cross-encoder) → **LB 0.963**

Run 4 was built from the run-3 error analysis. Removing every false merge was worth +0.8 validation
points, accepting every true match the model had rejected +1.6, and the candidate oracle was 0.985.

**What changed:**

| Change | Detail |
|---|---|
| Bigger India search | name/address/combined top-k 15/15/20 (was 10/10/15) + an **exact-name pass** (same normalised name, max 50 records per name). India test candidates before pruning: 60.4M pairs (S2+S3). |
| More training data | **304,555** training businesses (was 176,546): India 121,885, US 182,670 |
| New features | 16 run-4 features: duplicated-word and core-name checks, legal-form conflict, house-number suffix and 1-digit-edit checks, similarity to the best candidate, and group features from the pre-ranker (max, gap, share, count ≥ 0.5 …) |
| **Cross-encoder (e5)** | `intfloat/multilingual-e5-small` (**MIT licence, 118M parameters, pretrained**), fine-tuned on **1,023,262 labelled pairs** from 60,908 training businesses that the main model never trains on (clean stacking). It reads both records side by side and outputs a match probability. Training: laptop GPU (RTX 3050 6 GB, bf16), ~312 pairs/s, final loss ≈0.004. |
| Scoring with e5 | only the "uncertain" pairs (pre-ranker p in [0.02, 0.995]): **1.90M train + 13.28M test** pairs. Test scoring ran on **Kaggle 2×T4** (92 min). Kaggle and laptop scores agree to a mean difference of 0.0001. |

**Results:**

| | Run 3 | **Run 4** |
|---|---|---|
| Validation F0.5 (OOF) | 0.9613 | **0.9763** (t1 0.54, t2 0.74) |
| India | 0.9443 | **0.9671** (retrieval ceiling 0.9770) |
| US | 0.9726 | **0.9824** (retrieval ceiling 0.9941) |
| True matches retrieved (pair recall) | 0.9595 | 0.9657 |
| LightGBM validation log-loss per fold | 0.025–0.028 | **0.011** |
| Candidates per business | 18.0 | 22.6 |
| Public leaderboard | 0.941 | **0.963** (upload #3, 26 Sep 15:44) |
| France (worked out from the leaderboard) | ≈0.85 | **≈0.90** |

- The most important feature by far is the e5 score `ce_p`: about 3× the pre-ranker score and >10× every hand-made feature.
- Run 4 predicts 3.22 matches per business on average: France 3.00, India 3.20, US 3.34.
- The official validator passed with `--check-ids`.

## 10. Where the remaining errors are (run-4 analysis)

**Error budget on validation data:**

| | India | US |
|---|---|---|
| True matches **never retrieved** | 23,602 = **5.59%** | 11,004 = 1.74% |
| True matches dropped by pruning | 1,133 = 0.27% | 463 = 0.07% |
| Gain if every false merge were removed | +0.0035 | +0.0030 |
| Gain if every retrieved-but-rejected true match were accepted | +0.0064 | +0.0087 |

**Why were true matches never retrieved?** Missed pairs compared with found pairs:

| | India missed | India found | US missed | US found |
|---|---|---|---|---|
| Names near-identical after cleaning (similarity ≥ 90) | **75.8%** | 84.4% | **57.9%** | 82.9% |
| Candidate address empty | 19.2% | 3.1% | 45.8% | 4.2% |
| Candidate name in an Indic script | 34.0% | 17.3% | – | – |

**The main cause is crowding.** Names like "Surya Healthcare", "Blue Infra" or "Apex" exist hundreds
of times per country, so the right record falls out of the country-wide top-k name list. The city and
state words that would separate them are too common for the search index, so they get ignored.

**The noise is synthetic.** We see the same operations again and again:
- names rewritten in Indic scripts, with the address cut to "door no, city, state code" (`सिटी फूड्स प्राइवेट लिमिटेड ; 5, Mumbai, MH`);
- junk added to names: `***`, `--`, `Mr`, `(ID: 64721)`, or appended words (Service, Center, Partners…);
- look-alike characters: `R0OPESH`, `Internati0na1`;
- dropped or duplicated words (`Pvt Pvt Ltd`);
- the name replaced by gibberish at the same address.

**Name crowding per country** (share of test S2 records whose exact name is shared by more than 15 others):

| | Country-wide | Within the state/region |
|---|---|---|
| France | 19.5% | **12.7%** (only 3 regions) |
| India | 24.3% | 7.9% |
| US | 9.5% | 0.7% |

## 11. Team results so far

| Teammate | Task | Result |
|---|---|---|
| L2 | Decision layer on run-3 validation data | Baseline reproduced exactly (0.961270). Best variant: per-source thresholds S2 0.66 / S3 0.70 + 0.76 for rank ≥ 3, scoring 0.961311 on held-out folds. That is **+0.00004, which is noise**, and it was compared with the full-data baseline rather than the old rule on the same folds. Other methods: prob-sum −0.0028, expected-F0.5 −0.055. **Conclusion:** thresholds are exhausted. Only 0.2% of pairs fall between 0.65 and 0.75, so our errors are *confident* errors. Not merged. |
| L3 | GBDT tuning | not reported yet |
| L4 | mDeBERTa-v3-base (MIT, 280M) cross-encoder on Kaggle | training and scoring in progress |

## 12. Experiments that did **not** help (kept on record)

| Idea | Test | Result | Decision |
|---|---|---|---|
| **France pseudo-labelling** (retrain with our own confident French predictions) | Proxy with the US as the "unlabelled" country: India-only model 0.97782 on US; + pseudo-labels (99.74% correct) 0.97667 (**−0.0012**); looser variant 0.97684 (−0.0010) | worse | **dropped** |
| Decision-threshold tuning | L2, §11 | +0.00004 | not merged |
| City-level search for France | name crowding 12.7% (region) → ~9.8% (city, estimate) | small | not worth a re-run |

Side finding: a model trained only on India already scores 0.978 on the US, so the model transfers
across countries. France's weakness is therefore more likely its very generic names than its language.

## 13. Run 5 (running since 26 Sep 16:32)

| Change | Why | Evidence |
|---|---|---|
| **State/region-restricted name search** (k=10 per state) | fixes crowding (§10) | held-out true pairs: same region **98.4%** (India) / **93.1%** (US) when both are known; region found for 85% / 95% of pairs; France pools 96% |
| Exact-name pass for **all** countries (was India only) | US/France crowding | – |
| Retrieval-only name clean-up: look-alike digits (`0→o 1→l 3→e 4→a 5→s 7→t` inside words) and `(ID: n)` tags | synthetic noise (§10) | e.g. `internati0na1` → `international` |
| e5 band widened to [0.005, 0.995] | some rejected true matches had p < 0.02 and were never seen by e5 | run-4 e5 scores reused; only new pairs are scored |
| `candidate_pairs.tsv` keeps pairs with p ≥ 0.0001 | smaller candidate sets rank higher | run-4 validation: **22.6 → 7.3** candidates per business, true-match loss 0.003% |

Expected: validation above 0.9763 and a **leaderboard around 0.966–0.972**. This is an estimate, and
it will be confirmed or corrected by the validation score around 21:00. Run 5 is uploaded only if it
beats run 4 on validation.

## 14. Where everything is now, and what's next

| What | Where |
|---|---|
| Run-4 outputs (LB 0.963, current best) | `Downloads\PsychicLearn_run4_output\` |
| Run-5 work dir / outputs | `Downloads\PsychicLearn_work5\` / `Downloads\PsychicLearn_run5_output\` |
| Newest code | `Downloads\PsychicLearn_dev\src\`, repo `dev_run5/` |
| Files for teammates | GitHub release `from-L1` (run-3 and run-4 validation data, features, cross-encoder inputs) |
| Metrics and analyses | repo `results/` |

**Next:**
1. Run 5 validation → upload #4 if better.
2. L4's mDeBERTa scores → run 6 (`--ce-prefix ce2`) → upload if the validation gain is ≥ +0.0005.
3. 27 Sep: final package. The README must no longer claim "no pretrained models". Then the documentation, the zip, and a final upload of the best-validated file. Experiments stop at 20:00 IST.

The full plan is in `TEAM_PLAN.md` §7. Realistic target: **≈0.97–0.975**. 0.99 would need France near
0.99 without French labels; §10 and §12 explain why that is out of reach for us in the time left.

---

## Glossary

| Term | Meaning |
|---|---|
| **Entity resolution** | Deciding which records describe the same real-world thing |
| **Blocking / candidate generation** | A cheap first search that picks a short list of possible matches |
| **TF-IDF** | A way to turn text into numbers where rare words or letter-groups count more |
| **Recall ceiling / oracle** | The best score possible if the model were perfect on the candidates we found |
| **LightGBM** | A fast decision-tree learning algorithm (gradient boosting) |
| **Cross-validation (OOF)** | Scoring each training example with a model that didn't learn from it (honest validation) |
| **Transliteration** | Converting text from one script to another (Devanagari → Latin letters) |
| **Pruning / pre-ranker** | A small model that trims the candidate list before the main model |
| **Zero-shot** | Working on a category never seen in training (here: France) |
| **Cross-encoder** | A neural model that reads both records together and outputs a match probability (our e5 and mDeBERTa) |
| **Stacking** | Feeding one model's prediction into another model as a feature (e5 score → LightGBM) |
| **Pseudo-labelling** | Training on the model's own confident predictions for unlabelled data (tested for France; it hurt) |
| **Crowding** | The right record falls out of the top-k list because many other records have the same name |
| **Region pass** | Name search restricted to one state/region, to beat crowding (run 5) |

---

## 15. Run 6 and the finish (26–28 Sep)

| Upload | Time | What it was | Public leaderboard |
|---|---|---|---|
| #4 | 26 Sep, 23:45 | Run 5: region-restricted search, exact-name pass everywhere, name clean-up | **0.967** |
| #5 | 27 Sep, 03:18 | Run 6: + **dense retrieval** (L2) + **mDeBERTa cross-encoder** (L4) | **0.982608** (best) |

**Dense retrieval** was the breakthrough:
- `multilingual-e5-small` was fine-tuned as a bi-encoder on the training pairs, with the validation
  businesses excluded.
- On validation, it found 94.6% (India) and 82.3% (US) of the true matches that run 5's search missed.
- Search recall rose from 0.9688 to **0.9955**, and India's validation score from 0.969 to **0.988**.
- France (zero-shot) rose from about 0.918 to about **0.954**, worked out from the leaderboard.

**The finish:**
- The final code package (`submission/`, `SUBMISSIONS.md`) contains the run-6 outputs, the exact run-6
  code, a one-command reproduction `run_all.py` (tested end to end), pinned requirements, and the
  filled documentation.
- Team: Adarsh Singh (leader), Atharva Waghmode, Sanjog Poojary, Atharva Gadge.
