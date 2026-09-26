"""Run-4 extra features, appended to the pruned candidate files (stage "augment").

Error-analysis driven (run 3 validation):
* duplicated words ("Arihant Arihant Finance"): similarities on de-duplicated names
* dropped/changed house-number digits (1585 -> 585, 26695 -> 6695): suffix/prefix and 1-edit matches
* same-address look-alike businesses with a different legal form (Private Limited vs LLP): conflict flag
Group features (per Source-1 entity, from the pre-ranker probabilities of all its candidates):
* how many strong candidates the entity has, the gap to its best one, and how similar this candidate
  is to the entity's top candidate (records of one business look alike).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

GROUP = ["g_pre_max", "g_pre_gap", "g_n_pre50", "g_n_pre10", "g_pre_sum", "g_pre_share", "g_is_top"]
PAIR = ["xn_dedup_ratio", "xn_dedup_tset", "xn_core_equal", "x_legal_conflict", "xnum_suffix", "xnum_lev1",
        "xnum_subset", "g_top_name_sim", "g_top_addr_sim"]
EXTRA = GROUP + PAIR


def dedup_tokens(s):
    """Collapse consecutive repeated tokens ("arihant arihant finance" -> "arihant finance")."""
    out = []
    for t in s.split():
        if not out or out[-1] != t:
            out.append(t)
    return " ".join(out)


def group_features(s1_ids, pre_p, cand_ids):
    """Per-entity statistics of the pre-ranker probabilities; also returns each row's top candidate id."""
    g = pd.DataFrame({"s": s1_ids, "p": pre_p.astype(np.float32)})
    grp = g.groupby("s")["p"]
    f = pd.DataFrame(index=g.index)
    f["g_pre_max"] = grp.transform("max").values
    f["g_pre_gap"] = f["g_pre_max"].values - g["p"].values
    f["g_n_pre50"] = (g["p"] >= 0.5).astype(np.float32).groupby(g["s"]).transform("sum").values
    f["g_n_pre10"] = (g["p"] >= 0.1).astype(np.float32).groupby(g["s"]).transform("sum").values
    f["g_pre_sum"] = grp.transform("sum").values
    f["g_pre_share"] = g["p"].values / np.maximum(f["g_pre_sum"].values, 1e-6)
    top_idx = grp.transform("idxmax").values
    f["g_is_top"] = (top_idx == np.arange(len(g))).astype(np.float32)
    return f[GROUP].astype(np.float32), np.asarray(cand_ids, dtype=object)[top_idx]


def pair_features(s1_ids, cand_ids, top_ids, s1n, pooln, n_jobs=8):
    """String/number features for one batch of pairs (s1n/pooln indexed by entity_id)."""
    a, b, t = s1n.loc[s1_ids], pooln.loc[cand_ids], pooln.loc[top_ids]
    an = [dedup_tokens(x) for x in a["name_core"].values]
    bn = [dedup_tokens(x) for x in b["name_core"].values]
    tn = [dedup_tokens(x) for x in t["name_core"].values]
    f = pd.DataFrame(index=np.arange(len(an)))
    f["xn_dedup_ratio"] = process.cpdist(an, bn, scorer=fuzz.ratio, workers=n_jobs, dtype=np.float32)
    f["xn_dedup_tset"] = process.cpdist(an, bn, scorer=fuzz.token_set_ratio, workers=n_jobs, dtype=np.float32)
    f["xn_core_equal"] = (np.array(an, dtype=object) == np.array(bn, dtype=object)).astype(np.float32)
    f["x_legal_conflict"] = np.array([float(bool(x) and bool(y) and not (set(x.split()) & set(y.split())))
                                      for x, y in zip(a["legal"].values, b["legal"].values)], np.float32)
    an_, bn_, ap, bp = a["addr_nums"].values, b["addr_nums"].values, a["addr_primary"].values, b["addr_primary"].values
    suf = np.zeros(len(an), np.float32)
    lev1 = np.zeros(len(an), np.float32)
    sub = np.zeros(len(an), np.float32)
    for i in range(len(an)):
        pa, pb = ap[i], bp[i]
        if pa and pb and pa != pb and (pa.endswith(pb) or pb.endswith(pa) or pa.startswith(pb) or pb.startswith(pa)):
            suf[i] = 1.0
        sa, sb = an_[i].split(), bn_[i].split()
        if sa and sb:
            ssa, ssb = set(sa), set(sb)
            sub[i] = float(ssa <= ssb or ssb <= ssa)
            if not (ssa & ssb):
                lev1[i] = float(any(Levenshtein.distance(x, y) <= 1 for x in sa[:4] for y in sb[:4]))
    f["xnum_suffix"], f["xnum_lev1"], f["xnum_subset"] = suf, lev1, sub
    f["g_top_name_sim"] = process.cpdist(bn, tn, scorer=fuzz.token_set_ratio, workers=n_jobs, dtype=np.float32)
    f["g_top_addr_sim"] = process.cpdist(list(b["addr_core"].values), list(t["addr_core"].values),
                                         scorer=fuzz.token_set_ratio, workers=n_jobs, dtype=np.float32)
    return f[PAIR].astype(np.float32)
