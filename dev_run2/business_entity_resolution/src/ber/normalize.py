"""Name and address normalisation.

Every rule here targets a noise pattern measured in the training data (see the blueprint, §6.6):
injected accents, junk prefixes, website/handle forms, DBA ("<alias> trading as <name>"),
honorifics, legal-suffix variants (incl. dotted forms such as S.A.R.L.), street-type
abbreviations, zero-padded / suffixed house numbers, injected PO Box / PMB numbers and literal
"null" / "N/A" tokens. The rules are generic linguistic knowledge; no external data is used.
"""
import re
import unicodedata
from multiprocessing import Pool

import pandas as pd

LEGAL_CANON = {
    "pvt": "private", "private": "private", "prvt": "private",
    "ltd": "limited", "limited": "limited",
    "llc": "llc", "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llp": "llp", "lp": "lp", "pc": "pc", "pllc": "pllc", "plc": "plc", "public": "public", "opc": "opc",
    # French legal forms (zero-shot country in test)
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "sca": "sca", "scop": "scop",
}
HONORIFICS = {"mr", "mrs", "ms", "shri", "sri", "smt", "the"}
STOPWORDS = {"and", "of", "de", "du", "des", "la", "le", "les", "et", "d", "l"}
COUNTRY_WORDS = {"india", "france", "usa"}

DBA_RE = re.compile(r"\b(?:trading\s+as|t/a|a/k/a|aka|f/k/a|fka|formerly(?:\s+known\s+as)?|d/b/a|dba)\b", re.I)
DOTTED_ACRONYM_RE = re.compile(r"\b(?:[a-z]\.){2,}[a-z]?\.?")
TLD_RE = re.compile(r"\.(?:com|in|net|org|co|fr|biz|info)\b")
TOKEN_RE = re.compile(r"[a-z0-9À-ɏ]+|[ऀ-෿]+")
INDIC_RE = re.compile(r"[ऀ-෿]")

POBOX_RE = re.compile(r"\b(?:p\s*\.?\s*o\s*\.?\s*box|post\s*box|pmb)\s*[#:.\-]?\s*\d+")
NA_RE = re.compile(r"\bn\s*/\s*a\b")
ORD_RE = re.compile(r"^(\d+)(?:st|nd|rd|th)$")
NUM_ALPHA_RE = re.compile(r"^(\d+)([a-z]+)$")
ADDR_NULL = {"null", "none", "nan", "na", "no"}
ORDINAL_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
                 "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"}
STREET_ABBR = {
    "street": "st", "st": "st", "saint": "st", "road": "rd", "rd": "rd", "drive": "dr", "dr": "dr",
    "avenue": "ave", "ave": "ave", "av": "ave", "avenu": "ave", "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct", "circle": "cir", "cir": "cir", "boulevard": "blvd", "blvd": "blvd",
    "bd": "blvd", "parkway": "pkwy", "pkwy": "pkwy", "highway": "hwy", "hwy": "hwy",
    "trail": "trl", "trl": "trl", "place": "pl", "pl": "pl", "terrace": "ter", "turnpike": "tpke",
    "tpke": "tpke", "tpk": "tpke", "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt",
    "floor": "fl", "flr": "fl", "fl": "fl", "building": "bldg", "bldg": "bldg", "ngr": "nagar",
    "sec": "sector",
}
FRENCH_ABBR = {"r": "rue", "all": "allee", "imp": "impasse", "ch": "chemin", "fg": "faubourg",
               "fbg": "faubourg", "bis": "b", "rte": "route", "sq": "square"}


def latin_fold(text):
    """Lowercase + NFKC, then remove accents from Latin letters only.

    Combining marks attached to Indic letters are vowel signs, so they are kept.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    decomposed = unicodedata.normalize("NFKD", text)
    out, prev_latin = [], False
    for ch in decomposed:
        if unicodedata.combining(ch):
            if not prev_latin:
                out.append(ch)
            continue
        out.append(ch)
        prev_latin = ch < "ɐ"
    return unicodedata.normalize("NFC", "".join(out))


def name_views(name, country):
    """Return normalised views of a business name.

    Keys: core (tokens minus legal forms / honorifics / stopwords), nospace, acronym,
    legal (canonical legal forms, sorted), native (contains Indic script), handle (website /
    social-handle form), dba (had a trading-as style marker).
    """
    s = latin_fold(name)
    dba = False
    parts = DBA_RE.split(s)
    if len(parts) > 1 and parts[-1].strip():
        s, dba = parts[-1], True
    handle = bool(re.search(r"^[\s@#]*[@#]|www\.|\.(com|in|net|org|fr)\b", s))
    s = s.replace("www.", " ")
    s = TLD_RE.sub(" ", s)
    s = s.replace("m/s", " ")
    s = DOTTED_ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), s)
    s = s.replace("&", " and ").replace("+", " and ")
    toks = TOKEN_RE.findall(s)
    if len(toks) == 1 and toks[0].endswith("com") and len(toks[0]) > 6:  # e.g. "urologypartnerscom"
        toks, handle = [toks[0][:-3]], True
    cw = {country.lower()} | COUNTRY_WORDS
    core = [t for t in toks if t not in LEGAL_CANON and t not in HONORIFICS and t not in STOPWORDS and t not in cw]
    if not core:
        core = [t for t in toks if t not in HONORIFICS] or toks
    legal = sorted({LEGAL_CANON[t] for t in toks if t in LEGAL_CANON})
    latin_core = [t for t in core if not INDIC_RE.search(t)]
    return {
        "name_core": " ".join(core),
        "name_nospace": "".join(latin_core),
        "name_acr": "".join(t[0] for t in latin_core if t[0].isalpha()),
        "legal": " ".join(legal),
        "name_native": bool(INDIC_RE.search(s)),
        "name_handle": handle,
        "name_dba": dba,
        "name_ntok": len(core),
    }


def addr_views(addr, country):
    """Return normalised views of an address.

    Keys: addr_core (canonical tokens), addr_nums (distinct house/plot numbers, leading zeros
    stripped, ordinals and letter suffixes split off), addr_primary (first number), addr_empty.
    """
    s = latin_fold(addr)
    s = POBOX_RE.sub(" ", s)
    s = NA_RE.sub(" ", s)
    french = country == "France"
    out, nums = [], []
    for t in TOKEN_RE.findall(s):
        if t in ADDR_NULL:
            continue
        m = ORD_RE.match(t) or NUM_ALPHA_RE.match(t)
        if t.isdigit():
            t = str(int(t))
            nums.append(t)
        elif m:
            n = str(int(m.group(1)))
            nums.append(n)
            out.append(n)
            if m.re is NUM_ALPHA_RE and m.group(2) not in ("st", "nd", "rd", "th"):
                t = m.group(2)
            else:
                continue
        elif t in ORDINAL_WORDS:
            t = ORDINAL_WORDS[t]
            nums.append(t)
        elif french and t in FRENCH_ABBR:
            t = FRENCH_ABBR[t]
        else:
            t = STREET_ABBR.get(t, t)
        out.append(t)
    uniq = list(dict.fromkeys(nums))
    return {
        "addr_core": " ".join(out),
        "addr_nums": " ".join(uniq),
        "addr_primary": uniq[0] if uniq else "",
        "addr_empty": len(out) == 0,
        "addr_ntok": len(out),
    }


def _normalise_chunk(args):
    ids, names, addrs, countries = args
    rows = []
    for n, a, c in zip(names, addrs, countries):
        v = name_views(n, c)
        v.update(addr_views(a, c))
        rows.append(v)
    out = pd.DataFrame(rows)
    out.insert(0, "country", countries)
    out.insert(0, "entity_id", ids)
    return out


def _tasks(df, chunk):
    for i in range(0, len(df), chunk):
        yield (df["entity_id"].values[i:i + chunk], df["business_name"].values[i:i + chunk],
               df["business_address"].values[i:i + chunk], df["country"].values[i:i + chunk])


def normalise_frame(df, n_jobs=1, chunk=50_000):
    """Return entity_id, country and all name/address views (parallel over row chunks)."""
    return pd.concat(list(iter_normalised(df, n_jobs, chunk)), ignore_index=True)


def iter_normalised(df, n_jobs=1, chunk=50_000):
    """Yield normalised chunks in row order (lets callers stream them to disk)."""
    if n_jobs > 1 and len(df) > chunk:
        with Pool(n_jobs) as pool:
            yield from pool.imap(_normalise_chunk, _tasks(df, chunk))
    else:
        for t in _tasks(df, chunk):
            yield _normalise_chunk(t)
