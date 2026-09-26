# L4 guide: training the big cross-encoder on Kaggle

**Who this is for:** the L4 teammate (the RTX 4060 laptop). No Kaggle experience needed.

**Your job in one sentence:** fine-tune `microsoft/mdeberta-v3-base` on Kaggle's free GPUs so it learns
whether two business records are the same business, then use it to score the pairs Adarsh sends you.
Adarsh adds your scores to the main model as an extra feature.

**Why Kaggle and not your laptop:** this model did not fit in 6 GB. 8 GB is borderline and would be slow.
Kaggle gives two T4 GPUs (16 GB each) for free, and the job keeps running after you close the browser.

**Your time:** about 30 minutes of clicking. The rest runs unattended.

| Phase | When | Kaggle runtime (estimate) |
|---|---|---|
| 1. Train | as soon as `ce_train.parquet` is in release `from-L1` | about 1-2 h |
| 2. Score | when `score_train.parquet` + `score_test.parquet` appear in release `from-L1` | about 0.5-1 h |

The runtimes are estimates, because this model has never been run on a T4. The log shows the real ETA.

---

## Rules (please read)

- **Keep the Kaggle dataset PRIVATE.** It contains competition data, and making it public would break the
  competition rules.
- **Free tiers only.** Do not buy anything (no Colab Pro, no paid GPUs).
- **Don't change the script** except the `SMOKE` line and the fixes listed under Troubleshooting. If
  something else seems wrong, message Adarsh first.
- **Only upload to your own GitHub release:** `from-L4` (see Step 0 for how).

---

## Step 0: one-time setup (10 min)

1. Create an account at kaggle.com, or log in.
2. **Verify your phone number:** profile picture, then Settings, then Phone verification. Without this, the
   GPU and Internet options stay greyed out.
3. Accept Adarsh's invite to the private repo `Anexus5919/psychiclearn-amazon-ml-2026`. The email or
   notification comes from GitHub.
4. Get two files from the repo:
   - **`ce_train.parquet`:** repo page, then **Releases** (right sidebar), then **`from-L1`**, then click
     `ce_train.parquet` under *Assets*. It is about 1.02M labelled pairs, roughly 50-150 MB.
   - **The script:** the folder `kaggle/`, file `mdeberta_cross_encoder.py`. Open it and use the **Raw**
     button or the copy icon.

## Step 1: upload the training data as a private dataset

1. kaggle.com, then **Create**, then **New Dataset**.
2. Drag in `ce_train.parquet`.
3. Title: `psychiclearn-ce`. Check that visibility is **Private**. Click **Create**.
4. Wait until the dataset page shows the file (1-2 min).

## Step 2: create the notebook

1. kaggle.com, then **Create**, then **New Notebook**.
2. In the right-hand panel, open **Session options** (older UI: **Settings**):
   - **Accelerator: GPU T4 x2**. Not P100, and not "GPU T4" alone.
   - **Internet: On**. The script downloads the base model from Hugging Face.
3. In the right-hand panel, click **Add Input** (older UI: **Add Data**), open **Your Work**, then
   **Datasets**, and pick `psychiclearn-ce`.
4. Delete the example code in the first cell. Paste the **whole** script into that one cell.
5. Rename the notebook (top left) to `psychiclearn-mdeberta`.

## Step 3: smoke test (5 min, strongly recommended)

This catches problems in 5 minutes instead of 2 hours.

1. Near the top of the script, change `SMOKE = 0` to `SMOKE = 3000`.
2. Click **Run All** (the normal interactive run, not Save Version).
3. The output should show, in this order:
   - `GPUs: 2 ['Tesla T4', 'Tesla T4']`
   - `train pairs: 3,000  positives: 0.xxx`
   - a training progress bar
   - `training done in N min`
   - `ALL DONE  (SMOKE TEST - outputs are NOT usable ...)`
4. If you see an error instead, check the Troubleshooting section below or send Adarsh a screenshot.
5. **Set `SMOKE = 0` again.** This is important: a smoke-test model is useless.

## Step 4: the real training run (phase 1)

1. Make sure `SMOKE = 0`.
2. Top right: **Save Version**, then **Save & Run All (Commit)**, then **Save**.
3. You can now close the tab or shut down your laptop, because Kaggle runs the job on its servers.
4. To watch it: open the notebook, click the version number next to the title, then **Logs**.
   - Every 500 steps, a loss line appears. There are about 8,000 steps in total, and the loss should
     fall from about 0.3 towards 0.02-0.05.
   - If the loss shows `nan`, see Troubleshooting.
5. When it finishes, the log ends with `training done in N min` and `ALL DONE`.
6. **Write down** the training time and the last loss value, then post both in the team chat.
7. The version's **Output** tab now contains a `mdeberta_ce/` folder, about 1.1 GB. This is the trained
   model. **Don't delete this notebook version**, because phase 2 reuses it.

## Step 5: scoring run (phase 2)

Start this when Adarsh posts `score_train.parquet` and `score_test.parquet` in release `from-L1`.

1. Add the score files to your dataset:
   1. Open your dataset `psychiclearn-ce`.
   2. Click **New Version**.
   3. Upload both score files, and keep `ce_train.parquet`.
   4. Save.
2. Open the notebook `psychiclearn-mdeberta` in edit mode.
3. In the right-hand panel under Input, the dataset may say that a newer version exists. If so, click
   **update**, or remove the dataset and add it again.
4. Attach your trained model:
   1. Click **Add Input**.
   2. Open **Your Work**, then **Notebooks**.
   3. Pick `psychiclearn-mdeberta`, which is this same notebook's phase-1 output.
5. Check that `SMOKE = 0`. Then choose **Save Version**, then **Save & Run All (Commit)**.
6. **Check the first lines of the log.** One of them must say `SCORE-ONLY mode: using trained model at ...`.
   - If it starts training instead, the model input is not attached. Cancel the run and repeat step 4.
7. At the end, the log shows `scored train: ... pairs` and `scored test: ... pairs`, then `ALL DONE`.
8. From the version's **Output** tab, download `ce2_train.parquet` and `ce2_test.parquet`.
9. Upload both files to **your** release: repo page, then **Releases**, then **`from-L4`**, then the pencil
   icon (Edit), then drag the files into *Attach binaries*, then **Update release**. In the release
   description, write:
   - the phase-1 training time and final loss;
   - the phase-2 scoring time;
   - anything odd you noticed.

You're done with the Kaggle part. Tell Adarsh.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| GPU or Internet option greyed out | Verify your phone number (Step 0). |
| `AssertionError: attach the dataset with ce_train.parquet` | The dataset is not attached. Use Add Input and select `psychiclearn-ce`. |
| `OSError` / cannot download `microsoft/mdeberta-v3-base` | Internet is Off. Turn it on in Session options. |
| `GPUs: 1` or `GPUs: 0` | The accelerator is not **GPU T4 x2**. Change it and run again. |
| `CUDA out of memory` during training | In `TrainingArguments`, change `per_device_train_batch_size=64` to `32`. |
| `CUDA out of memory` during scoring | Change `def score(df, batch=512)` to `batch=256`. |
| Loss is `nan` | Tell Adarsh. The fix is `fp16=False`, but that roughly doubles the training time. |
| The run stops at 12 h | Kaggle's limit per run. This shouldn't happen; if it does, tell Adarsh. |
| GPU quota used up | Kaggle gives about 30 GPU hours per week, and each run uses 1-2 h. The quota is shown in the notebook editor. If it runs out, use Plan B. |

## Plan B: your own laptop (only if Kaggle fails)

This is slower and may not fit in 8 GB. Use it only if Kaggle is unavailable.

1. Install the packages: `pip install torch transformers pandas pyarrow sentencepiece protobuf`. The
   `torch` package must be the CUDA build from pytorch.org.
2. Put `ce_train.parquet`, and later the score files, in one folder, for example `C:\ber\in`.
3. In the script, change `per_device_train_batch_size=64` to `32`.
4. Run it:
   ```bat
   set BER_IN=C:\ber\in
   set BER_OUT=C:\ber\out
   python mdeberta_cross_encoder.py
   ```
   Phase 2 works the same way. Copy `C:\ber\out\mdeberta_ce` into `C:\ber\in\`, add the score files, and
   run the script again. It detects the model and only scores.
5. Keep the laptop plugged in and well ventilated. The GPU will run at 100% for hours.

## What the script does (for the curious)

- It reads each pair as `"name ; address"` of record A and record B, joined as a sentence pair and cut
  to 96 tokens.
- It fine-tunes mDeBERTa-v3-base (MIT licence, 280M parameters, allowed by the rules) for 1 epoch with a
  binary cross-entropy loss. It uses fp16 on both T4s.
- It saves the model to `mdeberta_ce/`.
- It then scores any `score_*.parquet` it finds and writes `ce2_*.parquet` with one match probability
  `ce_p` per pair.
- The training pairs come from businesses that the main model never trains on, so stacking the scores
  does not leak labels.
