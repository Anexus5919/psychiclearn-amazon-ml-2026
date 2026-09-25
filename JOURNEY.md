# PsychicLearn: What We Did, Start to Finish (so far)

Amazon ML Challenge 2026, Business Entity Resolution. Written 26 Sep 2026, around 03:00 IST.

| Upload | Time | What it was | Public leaderboard (F0.5) |
|---|---|---|---|
| #1 | 25 Sep, 23:59 | Run 2: first complete pipeline | **0.931** |
| #2 | 26 Sep, 02:50 | Run 3: + transliteration for India, + candidate pruning | **0.941** |

The leaderboard top was about 0.988 at the time of writing. Time remaining: about 45 hours (the deadline is 27 Sep, 23:59 IST).

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

Everything is written in Python. It uses **no external data, APIs or pretrained models**, which the fair-play rules require. The pipeline has 6 stages:

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

## 8. What's next (and how long it takes)

| Next step | Why | Time (coding + compute) |
|---|---|---|
| **Fix France** | Weakest country (~0.86–0.90) and 15% of the test set | ~1.5 h |
| **Train on more data** (8% → ~25%) | Sharper decisions on look-alike businesses | ~2 h |
| **Group-level features** (second-stage model using sibling records' scores) | Records of one business look alike, so decide them together | ~1.5–2 h |
| Possibly a smaller candidate budget (0.5% loss → ~12 per business) | Candidate size counts in the final ranking | ~30 min |
| Final package: zip + filled documentation + final upload | Required deliverable | ~1–1.5 h |

Realistic target: **0.96–0.975**. Getting to ~0.985 depends on how much of France and the group structure we can fix in the time left.

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
