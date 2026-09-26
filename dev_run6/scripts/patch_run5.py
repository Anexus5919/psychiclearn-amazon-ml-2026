import io, os, py_compile
B = r"C:\Users\ANEXUS\Downloads\PsychicLearn_dev\src\ber"


def patch(name, rep):
    p = os.path.join(B, name)
    s = io.open(p, encoding="utf-8").read()
    for a, b in rep:
        assert s.count(a) == 1, (name, a[:80])
        s = s.replace(a, b)
    io.open(p + ".new", "w", encoding="utf-8", newline="\n").write(s)
    py_compile.compile(p + ".new", doraise=True)
    return p


done = []
# ---------------------------------------------------------------- blocking.py
done.append(patch("blocking.py", [
('''import numpy as np
import pandas as pd
''',
'''import re

import numpy as np
import pandas as pd
'''),
('''PASSES = ("name", "addr", "combo")


def _texts(df, kind):
    if kind == "name":
        return df["name_core"].tolist()
    if kind == "addr":
        return df["addr_core"].tolist()
    # name tokens get an "n:" prefix so they never collide with address tokens
    return [" ".join("n:" + t for t in n.split()) + " " + a
            for n, a in zip(df["name_core"].tolist(), df["addr_core"].tolist())]
''',
'''PASSES = ("name", "addr", "combo")
REGION_PASS = len(PASSES) + 1  # pass id of the region-restricted name pass (the reverse pass is len(PASSES))

# retrieval-only name clean-up for synthetic noise seen in missed pairs: look-alike digits inside
# words ("internati0na1", "r0opesh") and injected record tags ("... (ID: 64721)")
_LEET = str.maketrans("013457", "oleast")
_LEET_TOK = re.compile(r"^(?=(?:.*[a-z]){2})(?=.*[013457])[a-z013457]+$")
_ID_TAG = re.compile(r"\\bid \\d+\\b")


def _clean_name(n):
    n = _ID_TAG.sub(" ", n)
    return " ".join(t.translate(_LEET) if _LEET_TOK.match(t) else t for t in n.split())


def _texts(df, kind):
    if kind == "name":
        return [_clean_name(n) for n in df["name_core"].tolist()]
    if kind == "addr":
        return df["addr_core"].tolist()
    # name tokens get an "n:" prefix so they never collide with address tokens
    return [" ".join("n:" + t for t in _clean_name(n).split()) + " " + a
            for n, a in zip(df["name_core"].tolist(), df["addr_core"].tolist())]
'''),
('''        log(f"    pass {kind:5s}: {len(rr):,} pairs")
''',
'''        log(f"    pass {kind:5s}: {len(rr):,} pairs")
        if kind == "name" and getattr(cfg, "k_region", 0) > 0 and "region" in s1.columns:
            # region-restricted name pass: same name TF-IDF, but each query only competes with pool
            # records of its own state/region (fixes country-wide "crowding" by same-named businesses)
            s1_reg, pool_reg = s1["region"].values, pool["region"].values
            pool_by = {r: g for r, g in pd.Series(np.arange(len(pool))).groupby(pool_reg)}
            q_reg = {r: g for r, g in pd.Series(q_rows).groupby(s1_reg[q_rows])}
            n_reg = 0
            for r, qs in q_reg.items():
                ps = pool_by.get(r)
                if not r or ps is None:
                    continue
                qr, pr = qs.values, ps.values
                ridx_, _ = topk(q[qr], p[pr], cfg.k_region, cfg.n_jobs)
                rr2, cc2 = np.nonzero(ridx_ >= 0)
                keys.append(qr[rr2].astype(np.int64) * n_pool + pr[ridx_[rr2, cc2]])
                passes.append(np.full(len(rr2), REGION_PASS, np.int8))
                ranks.append((cc2 + 1).astype(np.int16))
                n_reg += len(rr2)
            log(f"    pass region: {n_reg:,} pairs ({len(q_reg)} regions)")
'''),
('''    for pi, kind in enumerate(list(PASSES) + ["rev"]):
        k = cfg.k_reverse if kind == "rev" else ks[kind]
''',
'''    ks["rev"], ks["region"] = cfg.k_reverse, getattr(cfg, "k_region", 0)
    for pi, kind in enumerate(list(PASSES) + ["rev", "region"]):
        k = ks[kind]
'''),
('''    s_idx, p_idx = cand["s1"].values, cand["pool"].values
''',
'''    s_idx, p_idx = cand["s1"].values, cand["pool"].values
    if "region" in s1.columns and "region" in pool.columns:  # +1 same region, -1 different, 0 unknown
        ra, rb = s1["region"].values[s_idx], pool["region"].values[p_idx]
        known = (ra != "") & (rb != "")
        cand["region_match"] = np.where(known, np.where(ra == rb, 1, -1), 0).astype(np.int8)
    else:
        cand["region_match"] = np.int8(0)
'''),
]))
# ---------------------------------------------------------------- features.py
done.append(patch("features.py", [
('''    "cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "rank_rev",
    # competition''',
'''    "cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "rank_rev",
    "rank_region", "region_match",
    # competition'''),
('''    for col in ["cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "rank_rev",
                "rev_is_best",''',
'''    for col in ["cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "rank_rev",
                "rank_region", "region_match", "rev_is_best",'''),
]))
# ---------------------------------------------------------------- prune.py
done.append(patch("prune.py", [
('''CHEAP = ["cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "ctx_rank_combo_src",''',
'''CHEAP = ["cos_name", "cos_addr", "cos_combo", "rank_name", "rank_addr", "rank_combo", "rank_region", "region_match",
         "ctx_rank_combo_src",'''),
]))
# ---------------------------------------------------------------- pipeline.py
STAGE_REGIONS = '''def stage_regions(cfg):
    """State/region key per record (train+test, S1/S2/S3) for the region-restricted name pass."""
    from . import regions
    out = os.path.join(cfg.work_dir, "norm")
    paths = {(sp, src): os.path.join(out, f"{sp}_s{src}_region.parquet") for sp in ("train", "test") for src in (1, 2, 3)}
    if all(os.path.exists(p) for p in paths.values()):
        return
    dic = json.load(open(os.path.join(out, "translit.json"), encoding="utf-8"))["addr"]
    cols = ["entity_id", "business_address", "country"]
    s1 = pd.concat([io_utils.read_source(cfg.data_dir, sp, 1)[cols] for sp in ("train", "test")], ignore_index=True)
    vocab, cmap = regions.learn_s1(s1)
    log(f"regions: vocabulary {({c: len(v) for c, v in vocab.items()})}, S1 component map {({c: len(m) for c, m in cmap.items()})}", cfg)
    qpath = os.path.join(cfg.work_dir, "pairs", "train", "_queries.parquet")
    queries = set(pd.read_parquet(qpath)["s1_id"]) if os.path.exists(qpath) else train_query_ids(cfg, s1)
    gt = io_utils.read_tsv(os.path.join(cfg.data_dir, "train", "train_ground_truth.tsv"),
                           ["source1_entity_id", "matched_entity_ids"])
    gt = gt[gt["matched_entity_ids"] != ""]
    ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")[["source1_entity_id", "m"]]
    del gt
    s1a = s1.set_index("entity_id")["business_address"]
    learn, check = [], []
    for src in (2, 3):
        pool = io_utils.read_source(cfg.data_dir, "train", src)[cols]
        j = ex.merge(pool, left_on="m", right_on="entity_id")
        del pool
        held = j["source1_entity_id"].isin(queries).values
        for part, dst, n in ((j[~held], learn, 400_000), (j[held], check, 50_000)):
            part = part.sample(min(n, len(part)), random_state=0)
            dst += list(zip(part["country"].values, s1a.loc[part["source1_entity_id"]].values,
                            part["business_address"].values))
    gmap = regions.learn_pairs([(c, a, regions.components(b, dic)) for c, a, b in learn], vocab, cmap)
    log(f"regions: learned pool-component map (query entities excluded) {({c: len(m) for c, m in gmap.items()})}", cfg)
    agree = Counter()
    for c, a, b in check:  # held-out check on the query entities' true pairs
        ra = regions.region_of(regions.components(a), vocab[c], cmap[c])
        rb = regions.region_of(regions.components(b, dic), vocab[c], cmap[c], gmap.get(c, {}))
        agree[(c, "both" if ra and rb else "missing", ra == rb if ra and rb else None)] += 1
    log(f"regions: held-out true pairs (country, coverage, same region): {dict(agree)}", cfg)
    json.dump({"vocab": {c: sorted(v) for c, v in vocab.items()}, "s1_map": cmap, "pair_map": gmap},
              open(os.path.join(out, "regions.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for (sp, src), p in paths.items():
        if os.path.exists(p):
            continue
        df = io_utils.read_source(cfg.data_dir, sp, src)[cols]
        r = regions.assign(df, vocab, cmap, gmap, dic if src != 1 else None, cfg.n_jobs)
        cov = (r["region"] != "").groupby(df["country"].values).mean().round(3).to_dict()
        r.to_parquet(p, index=False)
        log(f"regions {sp} s{src}: coverage {cov}", cfg)


def load_norm(cfg, split, src, country=None, columns=None):'''
done.append(patch("pipeline.py", [
('''    exact_name_cap: int = 0         # >0 enables the exact core-name retrieval pass
''',
'''    exact_name_cap: int = 0         # >0 enables the exact core-name retrieval pass
    k_region: int = 0               # >0 enables the region-restricted name pass (see ber.regions)
'''),
('''def load_norm(cfg, split, src, country=None, columns=None):''', STAGE_REGIONS),
('''                s1 = load_norm(cfg, split, 1, country=country)
                pool = load_norm(cfg, split, src, country=country)
''',
'''                s1 = load_norm(cfg, split, 1, country=country)
                pool = load_norm(cfg, split, src, country=country)
                if cfg.k_region > 0:
                    for df_, s_ in ((s1, 1), (pool, src)):
                        reg = pd.read_parquet(os.path.join(cfg.work_dir, "norm", f"{split}_s{s_}_region.parquet"))
                        df_["region"] = df_["entity_id"].map(reg.set_index("entity_id")["region"]).fillna("").values
                        del reg
'''),
('''    ap.add_argument("--stage", default="all", choices=["prepare", "pairs",''',
'''    ap.add_argument("--stage", default="all", choices=["prepare", "regions", "pairs",'''),
('''                      ("k_reverse", int), ("exact_name_cap", int),''',
'''                      ("k_reverse", int), ("exact_name_cap", int), ("k_region", int),'''),
('''    for k in ("n_jobs", "train_frac", "k_name", "k_addr", "k_combo", "k_reverse", "exact_name_cap",''',
'''    for k in ("n_jobs", "train_frac", "k_name", "k_addr", "k_combo", "k_reverse", "exact_name_cap", "k_region",'''),
('''    if a.stage in ("pairs", "all"):
        stage_pairs(cfg)''',
'''    if a.stage == "regions" or (a.stage == "all" and cfg.k_region > 0):
        stage_regions(cfg)
    if a.stage in ("pairs", "all"):
        stage_pairs(cfg)'''),
('''import time
from dataclasses import asdict, dataclass
''',
'''import time
from collections import Counter
from dataclasses import asdict, dataclass
'''),
]))
for p in done:
    os.replace(p + ".new", p)
print("patched:", [os.path.basename(p) for p in done])
