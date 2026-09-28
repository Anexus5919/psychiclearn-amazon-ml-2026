# PsychicLearn - dense retrieval ("meaning search") on Kaggle (GPU T4 x2, Internet ON).
# Paste this whole file into ONE Kaggle notebook cell and run it.
#
# Fine-tunes intfloat/multilingual-e5-small (MIT licence, 118M parameters) as a bi-encoder on training
# ground-truth pairs (contrastive, in-batch negatives), embeds every record, and returns each Source-1
# record's K nearest Source-2/3 records (same split, same country) by cosine similarity.
# Why: run-4 retrieval missed 5.6% of India's true matches, mostly because of script changes and
# "crowding" by same-named businesses; similarity of learned embeddings is robust to both.
#
# Inputs (attach via "Add Input"; found anywhere under /kaggle/input):
#   the challenge dataset (train_source1..3.tsv, train_ground_truth.tsv, test_source1..3.tsv)
#   train_queries.parquet (s1_id): validation businesses. They are EXCLUDED from fine-tuning, so the
#   retrieval gain measured on them is honest.
# Outputs (/kaggle/working): dense_train.parquet, dense_test.parquet (s1_id, cand_id, dense_cos, dense_rank)
import glob, os, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
import pyarrow as pa, pyarrow.csv as pv
from transformers import AutoModel, AutoTokenizer

SMOKE = int(os.environ.get("BER_SMOKE", "0"))  # 1 = ~5-minute test on small samples; must be 0 for the real run
MODEL, MAXLEN, K = "intfloat/multilingual-e5-small", 64, 15
N_TRAIN_PAIRS, BATCH, LR, TAU = 2_000_000, 256, 3e-5, 0.05
OUT = os.environ.get("BER_OUT", "/kaggle/working")
IN = os.environ.get("BER_IN")
t00 = time.time()


def find(name):
    if IN:  # BER_IN may list several folders (dataset folder + inputs), separated by os.pathsep
        hits = sorted(h for r in IN.split(os.pathsep) for h in glob.glob(os.path.join(r, "**", name), recursive=True))
    else:
        hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    hits = [h for h in hits if "__MACOSX" not in h]
    assert hits, f"input file not found: {name} (attach the dataset)"
    return hits[0]


def read_tsv(path, cols):
    tb = pv.read_csv(path, parse_options=pv.ParseOptions(delimiter="\t"),
                     convert_options=pv.ConvertOptions(column_types={c: pa.string() for c in cols},
                                                       strings_can_be_null=False, null_values=[], include_columns=cols),
                     read_options=pv.ReadOptions(block_size=1 << 26))
    return tb.to_pandas()


def source(split, src):
    df = read_tsv(find(f"{split}_source{src}.tsv"), ["entity_id", "business_name", "business_address", "country"])
    df["text"] = "query: " + (df["business_name"] + " ; " + df["business_address"]).str.slice(0, 300)
    return df[["entity_id", "country", "text"]]


def log(m):
    print(f"[{(time.time() - t00) / 60:6.1f} min] {m}", flush=True)


n_gpu = torch.cuda.device_count()
log(f"GPUs: {n_gpu} {[torch.cuda.get_device_name(i) for i in range(n_gpu)]}")
tok = AutoTokenizer.from_pretrained(MODEL)
enc = AutoModel.from_pretrained(MODEL).cuda()
net = torch.nn.DataParallel(enc) if n_gpu > 1 else enc


def embed_batch(texts):
    b = tok(texts, truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
    b = {k: v.cuda() for k, v in b.items()}
    h = net(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1)


# ---------------------------------------------------------------- 1. fine-tune on training pairs
queries = set(pd.read_parquet(find("train_queries.parquet"))["s1_id"])
gt = read_tsv(find("train_ground_truth.tsv"), ["source1_entity_id", "matched_entity_ids"])
gt = gt[(gt["matched_entity_ids"] != "") & ~gt["source1_entity_id"].isin(queries)]
pairs = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")[["source1_entity_id", "m"]]
del gt
pairs = pairs.sample(min(N_TRAIN_PAIRS if not SMOKE else 20_000, len(pairs)), random_state=0)
s1 = source("train", 1).set_index("entity_id")["text"]
pool_text = pd.concat([source("train", 2), source("train", 3)]).set_index("entity_id")["text"]
A = s1.loc[pairs["source1_entity_id"]].values
B = pool_text.loc[pairs["m"]].values
del pairs, pool_text
log(f"fine-tuning on {len(A):,} pairs (validation businesses excluded: {len(queries):,})")
opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=0.01)
steps = len(A) // BATCH
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.05 * steps)) * max(0.0, (steps - s) / steps))
scaler = torch.cuda.amp.GradScaler()
net.train()
labels = torch.arange(BATCH, device="cuda")
for i in range(steps):
    sl = slice(i * BATCH, (i + 1) * BATCH)
    with torch.autocast("cuda", dtype=torch.float16):
        qa, qb = embed_batch(list(A[sl])), embed_batch(list(B[sl]))
        logits = (qa @ qb.T).float() / TAU
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()
    if i % 500 == 0:
        log(f"  step {i}/{steps} loss {loss.item():.4f}")
net.eval()
del A, B
enc.save_pretrained(f"{OUT}/dense_model")
tok.save_pretrained(f"{OUT}/dense_model")
log("fine-tuning done")


# ---------------------------------------------------------------- 2. embed + nearest neighbours
@torch.no_grad()
def embed_all(texts, batch=1024):
    order = np.argsort(np.fromiter((len(t) for t in texts), np.int32, len(texts)))
    out = torch.empty((len(texts), enc.config.hidden_size), dtype=torch.float16)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        with torch.autocast("cuda", dtype=torch.float16):
            out[torch.from_numpy(idx)] = embed_batch([texts[j] for j in idx]).half().cpu()
    return out


@torch.no_grad()
def knn(q, p, k, chunk=512):
    pg = p.cuda()
    idx = np.empty((len(q), k), np.int64)
    sc = np.empty((len(q), k), np.float32)
    for s in range(0, len(q), chunk):
        v, i = torch.topk(q[s:s + chunk].cuda() @ pg.T, k, dim=1)
        idx[s:s + chunk], sc[s:s + chunk] = i.cpu().numpy(), v.float().cpu().numpy()
    del pg
    torch.cuda.empty_cache()
    return idx, sc


for split in ("train", "test"):
    s1 = source(split, 1)
    if split == "train":
        s1 = s1[s1["entity_id"].isin(queries)]
    pool = pd.concat([source(split, 2), source(split, 3)], ignore_index=True)
    parts = []
    for country in sorted(s1["country"].unique()):
        q = s1[s1["country"] == country]
        p = pool[pool["country"] == country]
        if SMOKE:
            q, p = q.head(2000), p.head(50_000)
        log(f"{split} {country}: embedding {len(q):,} S1 + {len(p):,} S2/S3 records")
        qe, pe = embed_all(q["text"].tolist()), embed_all(p["text"].tolist())
        idx, sc = knn(qe, pe, K)
        parts.append(pd.DataFrame({
            "s1_id": np.repeat(q["entity_id"].values, K), "cand_id": p["entity_id"].values[idx.ravel()],
            "dense_cos": sc.ravel(), "dense_rank": np.tile(np.arange(1, K + 1, dtype=np.int16), len(q))}))
        log(f"  {split} {country}: {len(q) * K:,} pairs")
        del qe, pe
    pd.concat(parts, ignore_index=True).to_parquet(f"{OUT}/dense_{split}.parquet", index=False)
    log(f"wrote dense_{split}.parquet")
log("ALL DONE" + ("  (SMOKE TEST - outputs are NOT usable; set SMOKE = 0)" if SMOKE else ""))
