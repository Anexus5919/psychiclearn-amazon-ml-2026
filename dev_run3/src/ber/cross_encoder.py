"""Fine-tuned multilingual cross-encoder for pair matching (stacked into LightGBM as a feature).

The model reads both records together ("name ; address" of the Source-1 record and of the
candidate) and outputs a match probability. Base model: intfloat/multilingual-e5-small (MIT
licence, ~118M parameters), which covers the Latin, Devanagari and other Indic scripts in the data.
It is fine-tuned only on pairs of training entities that the LightGBM matcher never trains on.

Commands
  bench : measure train/inference throughput on this GPU
  train : fine-tune on <ce-work>/pairs_pruned/train
  score : score pairs (parquet with s1_id, cand_id) and write ce_p
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from . import io_utils

MODEL = "intfloat/multilingual-e5-small"


def raw_texts(data_dir, split, ids):
    """{entity_id: 'name ; address'} for the requested ids of one split (reads the raw TSVs)."""
    ids = set(ids)
    out = {}
    for src in (1, 2, 3):
        df = io_utils.read_source(data_dir, split, src)
        df = df[df["entity_id"].isin(ids)]
        out.update(dict(zip(df["entity_id"], (df["business_name"] + " ; " + df["business_address"]).str.slice(0, 300))))
    return out


class PairData(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return self.a[i], self.b[i], (self.y[i] if self.y is not None else 0.0)


def _collate(tok, max_len):
    def f(batch):
        a, b, y = zip(*batch)
        enc = tok(list(a), list(b), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(y, dtype=torch.float32)
        return enc
    return f


def train(a_txt, b_txt, y, out_dir, epochs=1, batch=64, lr=5e-5, max_len=96, log=print):
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
    dl = DataLoader(PairData(a_txt, b_txt, y), batch_size=batch, shuffle=True, collate_fn=_collate(tok, max_len),
                    num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(dl)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, done = time.time(), 0
    for ep in range(epochs):
        for i, enc in enumerate(dl):
            labels = enc.pop("labels").to(dev)
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**enc).logits.squeeze(-1)
            loss = lossf(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            done += len(labels)
            if i % 500 == 0:
                el = time.time() - t0
                log(f"  ep{ep} step {i}/{len(dl)} loss={loss.item():.4f} {done / max(el, 1e-9):.0f} pairs/s "
                    f"eta {(steps - i - ep * len(dl)) * batch / max(done / max(el, 1e-9), 1) / 60:.0f} min")
    log(f"TRAIN THROUGHPUT {done / max(time.time() - t0, 1e-9):.0f} pairs/s (excluding model load)")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


@torch.no_grad()
def score(a_txt, b_txt, model_dir, batch=512, max_len=96, log=print):
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(dev).eval()
    order = np.argsort([len(x) + len(y) for x, y in zip(a_txt, b_txt)])  # length-sorted batches: less padding
    out = np.zeros(len(a_txt), np.float32)
    t0 = time.time()
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([a_txt[i] for i in idx], [b_txt[i] for i in idx], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.squeeze(-1).float()
        out[idx] = torch.sigmoid(logits).cpu().numpy()
        if (s // batch) % 2000 == 0 and s:
            el = time.time() - t0
            log(f"  scored {s:,}/{len(order):,} ({s / el:.0f} pairs/s, eta {(len(order) - s) / (s / el) / 60:.0f} min)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "train", "score"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work", help="work dir holding pairs_pruned/<split>")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model-dir")
    ap.add_argument("--pairs", help="parquet with s1_id, cand_id to score")
    ap.add_argument("--out")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--model", default=MODEL, help="Hugging Face base model (MIT/Apache licensed)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()
    globals()["MODEL"] = a.model
    if a.cmd in ("bench", "train"):
        files = sorted(glob.glob(os.path.join(a.work, "pairs_pruned", "train", "[!_]*.parquet")))
        df = pd.concat([pd.read_parquet(f, columns=["s1_id", "cand_id"]) for f in files], ignore_index=True)
        gt = pd.read_parquet(os.path.join(a.work, "norm", "train_gt.parquet"))
        gt = {s: set(x.split(",")) if x else set() for s, x in zip(gt["s1_id"], gt["matched"])}
        df["y"] = [float(c in gt.get(s, ())) for s, c in zip(df["s1_id"], df["cand_id"])]
        if a.cmd == "bench":
            df = df.sample(20_000, random_state=0)
        elif a.max_pairs:
            df = df.sample(min(a.max_pairs, len(df)), random_state=0)
        txt = raw_texts(a.data_dir, "train", set(df["s1_id"]) | set(df["cand_id"]))
        at, bt = [txt[s] for s in df["s1_id"]], [txt[c] for c in df["cand_id"]]
        print(f"{len(df):,} pairs, positives {df['y'].mean():.3f}", flush=True)
        if a.cmd == "bench":
            t = time.time()
            train(at[:6400], bt[:6400], df["y"].values[:6400].tolist(), os.path.join(a.work, "ce_bench"), batch=a.batch,
                  log=lambda m: print(m, flush=True))
            tr = 6400 / (time.time() - t)
            t = time.time()
            score(at, bt, os.path.join(a.work, "ce_bench"))
            sc = len(at) / (time.time() - t)
            print(f"BENCH train {tr:.0f} pairs/s (incl. model load) | inference {sc:.0f} pairs/s", flush=True)
        else:
            train(at, bt, df["y"].values.tolist(), a.model_dir, epochs=a.epochs, batch=a.batch, lr=a.lr,
                  log=lambda m: print(m, flush=True))
            print("train done", flush=True)
    else:
        pairs = pd.read_parquet(a.pairs)
        txt = raw_texts(a.data_dir, a.split, set(pairs["s1_id"]) | set(pairs["cand_id"]))
        p = score([txt[s] for s in pairs["s1_id"]], [txt[c] for c in pairs["cand_id"]], a.model_dir,
                  log=lambda m: print(m, flush=True))
        pairs["ce_p"] = p
        pairs.to_parquet(a.out, index=False)
        print(f"scored {len(pairs):,} pairs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
