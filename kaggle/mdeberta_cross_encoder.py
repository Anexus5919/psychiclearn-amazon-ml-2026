# PsychicLearn - mDeBERTa-v3-base cross-encoder on Kaggle (GPU T4 x2, Internet ON).
# Paste this whole file into ONE Kaggle notebook cell and run it.
# Input dataset (attach via "Add Data"): a Kaggle dataset containing
#   ce_train.parquet   (text_a, text_b, label)            - required
#   score_train.parquet, score_test.parquet (s1_id, cand_id, text_a, text_b) - optional; scored if present
# Output (/kaggle/working): mdeberta_ce/ (model), ce_train.parquet / ce_test.parquet (s1_id, cand_id, ce_p)
# Model licence: microsoft/mdeberta-v3-base is MIT (allowed: MIT/Apache, <= 8B params).
import glob, os, time
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

MODEL, MAX_LEN = os.environ.get("BER_MODEL", "microsoft/mdeberta-v3-base"), 96
# Kaggle: files come from the attached dataset. Local GPU: set BER_IN / BER_OUT to folders.
IN = os.environ.get("BER_IN") or sorted(glob.glob("/kaggle/input/*/ce_train.parquet"))[0].rsplit("/", 1)[0]
OUT = os.environ.get("BER_OUT", "/kaggle/working")
os.makedirs(OUT, exist_ok=True)
print("GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
tok = AutoTokenizer.from_pretrained(MODEL)


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


tr = pd.read_parquet(f"{IN}/ce_train.parquet")
print(f"train pairs: {len(tr):,}  positives: {tr['label'].mean():.3f}")
t0 = time.time()
ds = Pairs(tr["text_a"], tr["text_b"], tr["label"].values)
print(f"tokenised in {time.time() - t0:.0f}s")
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, problem_type="regression")


class BCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels").float()
        out = model(**inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out.logits.squeeze(-1).float(), labels)
        return (loss, out) if return_outputs else loss


args = TrainingArguments(output_dir=f"{OUT}/ckpt", per_device_train_batch_size=64, learning_rate=3e-5,
                         num_train_epochs=1, warmup_ratio=0.05, weight_decay=0.01, fp16=True, logging_steps=500,
                         save_strategy="no", report_to=[], dataloader_num_workers=2, max_grad_norm=1.0)
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
    f = f"{IN}/score_{split}.parquet"
    if os.path.exists(f):
        df = pd.read_parquet(f)
        t1 = time.time()
        df["ce_p"] = score(df)
        df[["s1_id", "cand_id", "ce_p"]].to_parquet(f"{OUT}/ce_{split}.parquet", index=False)
        print(f"scored {split}: {len(df):,} pairs in {(time.time() - t1) / 60:.0f} min")
print("ALL DONE")
