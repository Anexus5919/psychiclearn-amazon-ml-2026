# L2 guide: dense retrieval ("meaning search") on Kaggle

**Who this is for:** L2, using your own Kaggle account. About 30 minutes of clicking, then ~2–2.5 h unattended.

**What it is.** Our pipeline first *searches* for ~20 possible matches per business, then a model *judges*
them. The judge is good. The search misses **5.6% of India's true matches** (run 4), because:
- names are written in another script (`सिटी फूड्स` vs `City Foods`);
- hundreds of businesses share the same name ("crowding").

This notebook fine-tunes a small multilingual model (`intfloat/multilingual-e5-small`, MIT licence,
118M parameters) so that records of the *same* business get similar "fingerprints". It then finds each
business's 15 nearest records by fingerprint. Adarsh adds these as extra candidates, and they are used
only if validation improves.

**Honest-evaluation rule:** the 304,555 validation businesses in `train_queries.parquet` are excluded
from fine-tuning, so the recall gain Adarsh measures on them is genuine.

## Step 1: get the files (10 min)
1. **Dataset zip:** download `6ab10eb3b23ba_student_resource.zip` (1.09 GB) from the challenge portal,
   **or** from the repo: **Releases → `from-L1`** → Assets.
2. **`train_queries.parquet`** (2 MB): repo **Releases → `from-L1`** → Assets.
3. **Script:** repo file `kaggle/dense_retrieval.py`. Open it and click the copy icon ("Copy raw file").

## Step 2: private Kaggle dataset (10–15 min)
1. kaggle.com → **+ Create → New Dataset**.
2. Drag in **both** the zip and `train_queries.parquet`. Don't unzip; Kaggle unpacks the zip itself.
3. Title `psychiclearn-raw`, visibility **Private**. It is competition data, so it must stay private.
4. Wait for both ticks, then click **Create**.
5. Wait until the dataset page lists the `student_resource` folder and `train_queries.parquet`.

## Step 3: notebook (5 min)
1. On the dataset page, click **`<> Code` → New Notebook**. The dataset is attached automatically.
2. **Settings → Accelerator → GPU T4 x2.**
3. **Settings → Internet → On.** The script downloads the base model.
4. Delete the sample code (Ctrl+A, Delete) and paste the whole script into one cell.

## Step 4: smoke test (~5–8 min, don't skip)
1. In the script, change `SMOKE = 0` to `SMOKE = 1`, then click **Run All**.
2. The output should show `GPUs: 2`, `fine-tuning on 20,000 pairs`, `fine-tuning done`, lines for
   train India/US and test France/India/US, and finally
   `ALL DONE (SMOKE TEST - outputs are NOT usable; set SMOKE = 0)`.
3. If you get an error, send Adarsh a screenshot.
4. Set it **back to `SMOKE = 0`** and click the **power icon** to stop the test session.

## Step 5: the real run (click, then walk away)
1. **Save Version** → name `dense run5` → **Save & Run All (Commit)** → **Save**.
2. After ~2 min the log shows `fine-tuning on 2,000,000 pairs`, then a loss line every 500 steps.
   The loss should fall, roughly from ~1–2 towards ~0.1–0.3.
3. You can close the browser. Total time is about **2–2.5 h**: ~30–45 min fine-tuning, then
   fingerprinting about 22M records and the nearest-neighbour search.
4. It ends with `wrote dense_test.parquet` and `ALL DONE`.

## Step 6: hand over (10 min)
1. Open the version's **Output** tab and download **`dense_train.parquet`** (~60–100 MB) and
   **`dense_test.parquet`** (~300–500 MB). `dense_model/` isn't needed.
2. Upload both to **your** release: repo → **Releases → `from-L2`** → pencil (Edit) → drag the files
   into *Attach binaries* → **Update release**.
3. Tell Adarsh.

## If something goes wrong
| Problem | Fix |
|---|---|
| `input file not found: train_source1.tsv` | The dataset isn't attached, or the zip wasn't unpacked. Check the Input panel. |
| `input file not found: train_queries.parquet` | Add `train_queries.parquet` to the dataset (New Version). |
| `GPUs: 1` or `0` | The accelerator isn't **GPU T4 x2**. |
| `CUDA out of memory` during fine-tuning | Change `BATCH = 256` to `128`. |
| `CUDA out of memory` in the neighbour search | Change `def knn(q, p, k, chunk=512)` to `chunk=256`. |
| Model download error | Internet is off. |

Rules: keep the dataset **Private**; free Kaggle only; don't change the script except `SMOKE` and the fixes above.
