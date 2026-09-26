"""Indic-script -> Latin transliteration learned from the provided training pairs.

About a quarter of matched India Source-2 names (and ~13% of Source-3 names) are written in an
Indic script while their Source-1 partner is Latin. Those ground-truth pairs form a parallel
corpus, so a token dictionary is learned from them (no external resources):

* names: when both names have the same number of tokens, tokens are aligned by position
  (e.g. "कृष्ण फाइनेंस प्राइवेट लिमिटेड" <-> "krishna finance private limited");
* addresses / leftovers: co-occurrence (a native token maps to the Latin token that appears in the
  largest share of its partners' Source-1 texts, weighted by rarity), e.g. "महाराष्ट्र" -> "maharashtra".

Tokens missing from the dictionary fall back to a generic romaniser built from Unicode character
names (e.g. "DEVANAGARI LETTER KA" -> "ka"), which works for every Indic script.
"""
import math
import re
import unicodedata
from collections import Counter, defaultdict

INDIC_TOKEN = re.compile(r"[ऀ-෿]+")
INDIC_CHAR = re.compile(r"[ऀ-෿]")
TOKEN_RE = re.compile(r"[a-z0-9À-ɏ]+|[ऀ-෿]+")


def _tokens(text):
    return TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())


def romanise(token):
    """Generic Indic -> Latin romanisation from Unicode character names (fallback only)."""
    out = []
    for ch in token:
        name = unicodedata.name(ch, "")
        if "DIGIT" in name:
            out.append(str(unicodedata.digit(ch, 0)))
        elif "VOWEL SIGN" in name:
            if out and out[-1].endswith("a") and len(out[-1]) > 1:
                out[-1] = out[-1][:-1]
            v = name.split("VOWEL SIGN ")[-1].lower().replace("vocalic ", "")
            out.append(v)
        elif "VIRAMA" in name:
            if out and out[-1].endswith("a") and len(out[-1]) > 1:
                out[-1] = out[-1][:-1]
        elif "ANUSVARA" in name or "CANDRABINDU" in name:
            out.append("n")
        elif "LETTER" in name:
            out.append(name.split("LETTER ")[-1].lower().replace("vocalic ", "").replace(" ", ""))
        # nukta, visarga, other signs: dropped
    s = "".join(out)
    return re.sub(r"([aeiou])\1+", r"\1", s)  # collapse long vowels (aa -> a)


def learn(pair_texts, min_count=2, min_share=0.6):
    """Learn {native_token: latin_token} from (latin_text, native_or_mixed_text) pairs.

    Positional alignment is used when both token lists have equal length; otherwise tokens feed a
    co-occurrence model. Returns the dictionary and simple stats.
    """
    pos = defaultdict(Counter)
    cooc = defaultdict(Counter)
    native_df = Counter()
    latin_df = Counter()
    n_pairs = 0
    for latin_text, other_text in pair_texts:
        lt, ot = _tokens(latin_text), _tokens(other_text)
        if not any(INDIC_CHAR.search(t) for t in ot) or any(INDIC_CHAR.search(t) for t in lt):
            continue
        n_pairs += 1
        lset = set(lt)
        for t in lset:
            latin_df[t] += 1
        if len(lt) == len(ot):
            for a, b in zip(ot, lt):
                if INDIC_CHAR.search(a):
                    pos[a][b] += 1
        for a in set(ot):
            if INDIC_CHAR.search(a):
                native_df[a] += 1
                for b in lset:
                    cooc[a][b] += 1
    dic = {}
    for a, c in pos.items():
        b, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_share:
            dic[a] = b
    for a, c in cooc.items():
        if a in dic or native_df[a] < min_count:
            continue
        best, best_score = None, 0.0
        for b, n in c.items():
            share = n / native_df[a]
            if share < min_share:
                continue
            score = share * math.log(1 + n_pairs / latin_df[b])
            if score > best_score:
                best, best_score = b, score
        if best is not None:
            dic[a] = best
    return dic, {"pairs_used": n_pairs, "positional": sum(1 for a in pos if a in dic), "entries": len(dic)}


def apply(text, dic):
    """Replace native-script tokens in `text` by their learned (or romanised) Latin form."""
    if not INDIC_CHAR.search(text):
        return text
    return INDIC_TOKEN.sub(lambda m: " " + dic.get(m.group(0), romanise(m.group(0))) + " ", text)
