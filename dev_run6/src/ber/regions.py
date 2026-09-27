"""Coarse location key (state / region) per record, for region-restricted name retrieval.

Why: in run 4, 76% (India) and 58% (US) of the true matches that retrieval missed had near-identical
normalised names. The country-wide top-k name lists were full of same-named businesses in other
states ("crowding"). Searching names within one state/region removes most of that competition.

S1 addresses end with the region (India: state, US: state code, France: region). Pool records often
abbreviate it (MH, KA), write it in an Indic script, spell out the state (ohio), give a French
departement, or only a city. The maps are built from the provided files only:
  * label-free, from S1 addresses: the region vocabulary (frequent last components that rarely occur
    elsewhere in an address) and a component -> region map for every other S1 component (cities,
    districts), used for all countries including France (no French labels exist);
  * from training ground-truth pairs (query entities excluded): pool address component -> region
    of the matched S1 (learns abbreviations such as "mh" -> "maharashtra", per country).
"""
import re
from collections import Counter, defaultdict

import pandas as pd

from . import translit
from .normalize import latin_fold

SPLIT_RE = re.compile(r"[,;]")
CLEAN_RE = re.compile(r"[^a-z0-9ऀ-෿]+")


def components(addr, dic=None):
    """Address split at commas; each part folded, transliterated and cleaned."""
    if not isinstance(addr, str) or not addr:
        return []
    if dic and translit.INDIC_CHAR.search(addr):
        addr = translit.apply(addr, dic)
    out = []
    for part in SPLIT_RE.split(addr):
        c = CLEAN_RE.sub(" ", latin_fold(part)).strip()
        if c:
            out.append(c)
    return out


def learn_s1(s1, min_last=300, min_last_share=0.8, min_city=3, min_city_share=0.95):
    """Region vocabulary and component->region map per country, from S1 addresses (no labels)."""
    vocab, cmap = {}, {}
    for country, g in s1.groupby("country"):
        comps = [components(a) for a in g["business_address"].values]
        last, other = Counter(), Counter()
        for cs in comps:
            if cs:
                last[cs[-1]] += 1
                other.update(cs[:-1])
        voc = {c for c, n in last.items() if n >= min_last and n / (n + other[c]) >= min_last_share}
        cnt = defaultdict(Counter)
        for cs in comps:
            r = next((c for c in reversed(cs) if c in voc), None)
            if r is None:
                continue
            for c in cs:
                if c not in voc:
                    cnt[c][r] += 1
        m = {}
        for c, rc in cnt.items():
            r, n = rc.most_common(1)[0]
            tot = sum(rc.values())
            if tot >= min_city and n / tot >= min_city_share:
                m[c] = r
        vocab[country], cmap[country] = voc, m
    return vocab, cmap


def region_of(cs, voc, cmap, gmap=None):
    """Region of one record from its address components (scanned from the last one backwards)."""
    for c in reversed(cs):
        if c in voc:
            return c
    for c in reversed(cs):
        r = cmap.get(c) or (gmap.get(c) if gmap else None)
        if r:
            return r
    if cs:  # region glued to the last component, e.g. "rocford illinois" / "mumbai mh"
        toks = cs[-1].split()
        for n in (2, 1):
            if len(toks) > n:
                t = " ".join(toks[-n:])
                r = t if t in voc else (cmap.get(t) or (gmap.get(t) if gmap else None))
                if r:
                    return r
    return ""


def learn_pairs(pairs, vocab, cmap, min_count=20, min_share=0.9):
    """pool component -> S1 region, learned from (country, s1_address, pool_address_components) pairs."""
    cnt = defaultdict(lambda: defaultdict(Counter))
    for country, s1_addr, pcs in pairs:
        r = region_of(components(s1_addr), vocab.get(country, set()), cmap.get(country, {}))
        if not r:
            continue
        for c in pcs:
            cnt[country][c][r] += 1
    gmap = {}
    for country, d in cnt.items():
        m = {}
        for c, rc in d.items():
            r, n = rc.most_common(1)[0]
            tot = sum(rc.values())
            if tot >= min_count and n / tot >= min_share:
                m[c] = r
        gmap[country] = m
    return gmap


_STATE = None


def _init(state):
    global _STATE
    _STATE = state


def _regions_chunk(args):
    addrs, countries = args
    vocab, cmap, gmap, dic = _STATE
    return [region_of(components(a, dic), vocab.get(c, set()), cmap.get(c, {}), gmap.get(c, {}))
            for a, c in zip(addrs, countries)]


def assign(df, vocab, cmap, gmap, dic, n_jobs=1, chunk=100_000):
    """Region for every row of a raw source frame (entity_id, business_address, country)."""
    from multiprocessing import Pool
    tasks = [(df["business_address"].values[i:i + chunk], df["country"].values[i:i + chunk])
             for i in range(0, len(df), chunk)]
    state = (vocab, cmap, gmap, dic)
    if n_jobs > 1 and len(tasks) > 1:
        with Pool(n_jobs, initializer=_init, initargs=(state,)) as pool:
            parts = pool.map(_regions_chunk, tasks)
    else:
        _init(state)
        parts = [_regions_chunk(t) for t in tasks]
    return pd.DataFrame({"entity_id": df["entity_id"].values, "region": [r for p in parts for r in p]})
