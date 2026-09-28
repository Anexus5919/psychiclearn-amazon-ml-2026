# PsychicLearn - mDeBERTa-v3-base cross-encoder on Kaggle (GPU T4 x2, Internet ON).
# Paste this whole file into ONE Kaggle notebook cell and run it.
# Inputs (attach via "Add Data"; found anywhere under /kaggle/input):
#   ce_train.parquet   (text_a, text_b, label)                                  - phase 1 (training)
#   score_train.parquet, score_test.parquet (s1_id, cand_id, text_a, text_b)     - scored if present
#   mdeberta_ce/ or ce_model/ (a trained model, e.g. phase-1 notebook output)   - if present: SCORE-ONLY, no training
# Output (/kaggle/working): mdeberta_ce/ (model), ce2_train.parquet / ce2_test.parquet (s1_id, cand_id, ce_p)
# Model licence: microsoft/mdeberta-v3-base is MIT (allowed: MIT/Apache, <= 8B params).
import glob, os, time
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

SMOKE = int(os.environ.get("BER_SMOKE", "0"))  # e.g. 3000 = ~3-minute test on a tiny sample; must be 0 for the real run
MODEL, MAX_LEN = os.environ.get("BER_MODEL", "microsoft/mdeberta-v3-base"), 96
# Kaggle: files come from the attached inputs. Local GPU: set BER_IN / BER_OUT to folders.
IN, OUT = os.environ.get("BER_IN"), os.environ.get("BER_OUT", "/kaggle/working")


def find(rel):
    """Path of an input file/folder: BER_IN if set, else searched anywhere under /kaggle/input."""
    if IN:
        return os.path.join(IN, rel) if os.path.exists(os.path.join(IN, rel)) else None
    hits = sorted(glob.glob(f"/kaggle/input/**/{rel}", recursive=True))
    return hits[0] if hits else None


os.makedirs(OUT, exist_ok=True)
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


class Pairs(Dataset):
    def __init__(self, a, b, y=None):
        self.enc = tok(list(a), list(b), truncation=True, max_length=MAX_LEN)
        self.y = y

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.y is not None:
            item["labels"] = float(self.y[i])
        return item


# Phase 2 (score-only): if an already-trained model is attached as input (a folder "mdeberta_ce" with
# config.json, e.g. the output of the phase-1 notebook added as a dataset), skip training entirely.
trained = find("mdeberta_ce/config.json") or find("ce_model/config.json")
t0 = time.time()
if trained:
    MODEL_DIR = os.path.dirname(trained)
    print("SCORE-ONLY mode: using trained model at", MODEL_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
else:
    assert find("ce_train.parquet"), "attach the dataset with ce_train.parquet (or a trained mdeberta_ce/ or ce_model/)"
    tok = AutoTokenizer.from_pretrained(MODEL)
    tr = pd.read_parquet(find("ce_train.parquet"))
    if SMOKE:
        tr = tr.sample(min(SMOKE, len(tr)), random_state=0).reset_index(drop=True)
    print(f"train pairs: {len(tr):,}  positives: {tr['label'].mean():.3f}")
    ds = Pairs(tr["text_a"], tr["text_b"], tr["label"].values)
    print(f"tokenised in {time.time() - t0:.0f}s")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, problem_type="regression")


class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels").float()
        out = model(**inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out.logits.squeeze(-1).float(), labels)
        return (loss, out) if return_outputs else loss


if not trained:  # phase 1: fine-tune and save the model (phase 2 reuses it as an input dataset)
    args = TrainingArguments(output_dir=f"{OUT}/ckpt", per_device_train_batch_size=64, learning_rate=3e-5,
                             num_train_epochs=1, warmup_ratio=0.05, weight_decay=0.01, fp16=True, logging_steps=500,
                             save_strategy="no", report_to=[], dataloader_num_workers=0 if os.name == "nt" else 2, max_grad_norm=1.0)
    BCETrainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorWithPadding(tok)).train()
    model.save_pretrained(f"{OUT}/mdeberta_ce")
    tok.save_pretrained(f"{OUT}/mdeberta_ce")
    print(f"training done in {(time.time() - t0) / 60:.0f} min")


@torch.no_grad()
def score(df, batch=512):
    model.eval().cuda()
    m = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    order = np.argsort((df["text_a"].str.len() + df["text_b"].str.len()).values)
    out = np.zeros(len(df), np.float32)
    a, b = df["text_a"].values, df["text_b"].values
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok(list(a[idx]), list(b[idx]), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        enc = {k: v.cuda() for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            out[idx] = torch.sigmoid(m(**enc).logits.squeeze(-1).float()).cpu().numpy()
    return out


for split in ("train", "test"):
    f = find(f"score_{split}.parquet")
    if f:
        df = pd.read_parquet(f)
        if SMOKE:
            df = df.head(SMOKE)
        t1 = time.time()
        df["ce_p"] = score(df)
        df[["s1_id", "cand_id", "ce_p"]].to_parquet(f"{OUT}/ce2_{split}.parquet", index=False)
        print(f"scored {split}: {len(df):,} pairs in {(time.time() - t1) / 60:.0f} min")
print("ALL DONE" + ("  (SMOKE TEST - outputs are NOT usable; set SMOKE = 0 for the real run)" if SMOKE else ""))
