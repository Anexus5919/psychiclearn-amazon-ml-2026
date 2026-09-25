"""Measure recall ceiling of simple char-3gram TF-IDF top-K blocking (name / address / union)
against the FULL same-country train S2/S3 pools, plus a naive no-ML threshold baseline F0.5."""
import gc
import re
import sys
import time
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

torch.set_num_threads(12)
torch.sparse.check_sparse_tensor_invariants.enable()  # fail loudly instead of silently wrong
BASE = r"C:\Users\ANEXUS\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"
COLS = ["entity_id", "business_name", "business_address", "country"]
NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
KMAX = 50
rng = np.random.default_rng(0)


def read(path, cols):
    return pv.read_csv(
        path, parse_options=pv.ParseOptions(delimiter="\t"),
        convert_options=pv.ConvertOptions(column_types={c: pa.string() for c in cols},
                                          strings_can_be_null=False, null_values=[], include_columns=cols),
        read_options=pv.ReadOptions(block_size=1 << 26))


DROP = set("""pvt private ltd limited llc inc incorporated corp corporation co company llp lp pc pllc plc
public the and of mr mrs ms m s shri sri smt""".split())
DBA = re.compile(r"\b(?:trading as|t/a|a/k/a|aka|f/k/a|fka|formerly|d/b/a|dba)\b", re.I)
ABBR = {"st": "street", "rd": "road", "dr": "drive", "ave": "avenue", "av": "avenue", "ln": "lane",
        "ct": "court", "cir": "circle", "blvd": "boulevard", "pkwy": "parkway", "hwy": "highway",
        "trl": "trail", "pl": "place", "ter": "terrace", "tpke": "turnpike"}


def fold(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def norm_name(s):
    parts = DBA.split(s)
    if len(parts) > 1:
        s = parts[-1]
    s = fold(s)
    s = re.sub(r"\.(com|in|net|org|co|fr)\b", " ", s)
    s = s.replace("&", " ")
    toks = [t for t in re.split(r"[^0-9a-z\u0900-\u0dff]+", s) if t and t not in DROP]
    return " ".join(toks)


def norm_addr(s):
    s = fold(s)
    s = re.sub(r"\b(p\.?\s?o\.?\s?box|pmb)\s*\d+", " ", s)
    out = []
    for t in re.split(r"[^0-9a-z\u0900-\u0dff]+", s):
        if not t or t in ("null", "n", "a", "na"):
            continue
        if t.isdigit():
            t = t.lstrip("0") or "0"
        out.append(ABBR.get(t, t))
    return " ".join(out)


def topk_search(pool_texts, q_texts):
    """Cosine top-K over pool (chunked, CPU torch). Returns scores, idx and the query-vocab-restricted
    matrices Pr/Qr; because rows were L2-normalised over the full vocab, Pr[j].Qr[r] is the exact cosine."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
    P = vec.fit_transform(pool_texts).tocsr()
    Q = vec.transform(q_texts).tocsr()
    # BUG FIXED (invalidated runs v1/v2): Q used to be CSC, whose .indices are ROW indices, so the
    # "query vocabulary" was really feature ids 0..nq-1. Columns must come from a CSR matrix.
    cols = np.unique(Q.indices)
    Pr, Qr = P[:, cols].tocsr(), Q[:, cols].tocsr()
    Pr.sort_indices()  # torch CSR expects sorted per-row column indices
    Qr.sort_indices()
    Qt = torch.from_numpy(Qr.toarray().T.copy())
    best_s = torch.full((KMAX, Qt.shape[1]), -1.0)
    best_i = torch.zeros((KMAX, Qt.shape[1]), dtype=torch.int64)
    CH = 60000
    for s in range(0, Pr.shape[0], CH):
        c = Pr[s:s + CH]
        t = torch.sparse_csr_tensor(torch.from_numpy(c.indptr.astype(np.int64)),
                                    torch.from_numpy(c.indices.astype(np.int64)),
                                    torch.from_numpy(c.data), size=c.shape)
        v, i = torch.topk(t @ Qt, min(KMAX, c.shape[0]), dim=0)
        v2, j = torch.topk(torch.cat([best_s, v]), KMAX, dim=0)
        best_i = torch.gather(torch.cat([best_i, i + s]), 0, j)
        best_s = v2
    bs, bi = best_s.T.numpy(), best_i.T.numpy()
    # INDEPENDENT self-check: exact brute force on the FULL (unrestricted) matrices for 10 queries.
    # Checks the K-th best score AND that the returned indices really have the returned scores.
    PT = P.T.tocsr()
    for r in range(10):
        col = (Q[r] @ PT).toarray().ravel()
        exact_kth = np.partition(col, -KMAX)[-KMAX]
        assert abs(exact_kth - bs[r, -1]) < 1e-4, ("kth", r, exact_kth, bs[r, -1])
        assert np.allclose(col[bi[r]], bs[r], atol=1e-4), ("idx", r)
    del P, Q, PT
    return bs, bi, Pr, Qr


def rowdot(A, B):
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


def fbeta(pred, true, b2=0.25):
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + b2) * p * r / (b2 * p + r)


gt = read(f"{BASE}\\train\\train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
gtd = dict(zip(gt.column(0).to_pylist(), gt.column(1).to_pylist()))
del gt
s1 = read(f"{BASE}\\train\\train_source1.tsv", COLS)

for country in ("US", "India"):
    t0 = time.time()
    sub = s1.filter(pc.equal(s1.column("country"), country))
    q = sub.take(pa.array(rng.choice(sub.num_rows, NQ, replace=False))).to_pylist()
    qids = [r["entity_id"] for r in q]
    truth = {r["entity_id"]: {x for x in (gtd[r["entity_id"]] or "").split(",") if x} for r in q}
    qn = [norm_name(r["business_name"]) for r in q]
    qa = [norm_addr(r["business_address"]) for r in q]
    cand = {i: {} for i in qids}  # s1 -> {cand_id: (name_cos, addr_cos)}
    rec = {}
    for src in ("2", "3"):
        pool = read(f"{BASE}\\train\\train_source{src}.tsv", COLS)
        pool = pool.filter(pc.equal(pool.column("country"), country))
        pids = np.array(pool.column("entity_id").to_pylist(), dtype=object)
        pn = [norm_name(x) for x in pool.column("business_name").to_pylist()]
        pad = [norm_addr(x) for x in pool.column("business_address").to_pylist()]
        del pool
        gc.collect()
        ns, ni, Pn, Qn = topk_search(pn, qn)
        as_, ai, Pa, Qa = topk_search(pad, qa)
        del pn, pad
        for K in (5, 10, 20, 50):
            hn = ha = hu = tot = 0
            for r, qid in enumerate(qids):
                tset = {m for m in truth[qid] if m.startswith(f"S{src}-")}
                if not tset:
                    continue
                cn, ca = set(pids[ni[r, :K]]), set(pids[ai[r, :K]])
                tot += len(tset); hn += len(tset & cn); ha += len(tset & ca); hu += len(tset & (cn | ca))
            rec[(src, K)] = (hn / tot, ha / tot, hu / tot)
        rows, cols_ = [], []
        for r in range(len(qids)):
            for j in set(ni[r, :20].tolist()) | set(ai[r, :20].tolist()):
                rows.append(r); cols_.append(j)
        rows, cols_ = np.array(rows), np.array(cols_)
        nc, ac = rowdot(Pn[cols_], Qn[rows]), rowdot(Pa[cols_], Qa[rows])
        for r, j, x, y in zip(rows, cols_, nc, ac):
            cand[qids[r]][pids[j]] = (float(x), float(y))
        del Pn, Pa, Qn, Qa, pids
        gc.collect()
        print(f"[{country}] S{src} done at {time.time() - t0:.0f}s", flush=True)

    print(f"\n=== {country}: recall of true S{{2,3}} matches within top-K (name / address / union), {NQ} queries")
    for (src, K), (a, b, c) in sorted(rec.items()):
        print(f"  S{src} K={K:>2}: name={a:.4f}  addr={b:.4f}  union={c:.4f}")
    tot = sum(len(v) for v in truth.values())
    u20 = sum(len(truth[qid] & set(cand[qid])) for qid in qids)
    print(f"  S2+S3 name@20 U addr@20 overall pair recall: {u20 / tot:.4f}   "
          f"mean candidates per S1: {np.mean([len(cand[x]) for x in qids]):.1f}")
    best = (0, None)
    for w in np.linspace(0, 1, 11):
        for t in np.linspace(0.1, 0.98, 45):
            f = np.mean([fbeta({c for c, (x, y) in cand[qid].items() if w * x + (1 - w) * y >= t}, truth[qid])
                         for qid in qids])
            if f > best[0]:
                best = (f, (round(w, 2), round(t, 3)))
    print(f"  naive no-ML baseline (predict if w*name_cos+(1-w)*addr_cos >= t, tuned on these queries): "
          f"macro F0.5={best[0]:.4f} at (w,t)={best[1]}")
    oracle = np.mean([fbeta(truth[qid] & set(cand[qid]), truth[qid]) for qid in qids])
    print(f"  oracle F0.5 given these candidates (perfect classifier): {oracle:.4f}\n", flush=True)
