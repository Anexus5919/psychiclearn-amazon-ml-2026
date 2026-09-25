"""Ground-truth structure, coverage, leakage and train/test overlap checks."""
import gc
import random

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

BASE = r"C:\Users\ANEXUS\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset"


def read(path, cols):
    return pv.read_csv(
        path,
        parse_options=pv.ParseOptions(delimiter="\t"),
        convert_options=pv.ConvertOptions(
            column_types={c: pa.string() for c in cols},
            strings_can_be_null=False, null_values=[], include_columns=cols),
        read_options=pv.ReadOptions(block_size=1 << 26),
    )


def col(tb, c):
    return tb.column(c).combine_chunks()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


gt = read(f"{BASE}\\train\\train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
g1 = col(gt, "source1_entity_id")
gm = col(gt, "matched_entity_ids")
n = len(g1)
print(f"GT rows: {n:,}  unique S1: {pc.count_distinct(g1).as_py():,}")

lists = pc.split_pattern(gm, ",")
lens = pc.list_value_length(lists).to_numpy(zero_copy_only=False)
empty = pc.equal(pc.utf8_trim_whitespace(gm), "").to_numpy(zero_copy_only=False)
lens = np.where(empty, 0, lens)
flat = pc.list_flatten(lists)
parent = pc.list_parent_indices(lists).to_numpy()
keep = pc.not_equal(flat, "")
flat = pc.filter(flat, keep)
parent = parent[keep.to_numpy(zero_copy_only=False)]
is2 = pc.starts_with(flat, "S2-").to_numpy(zero_copy_only=False)
is3 = pc.starts_with(flat, "S3-").to_numpy(zero_copy_only=False)
n2 = np.bincount(parent[is2], minlength=n)
n3 = np.bincount(parent[is3], minlength=n)

print(f"\nsingletons (empty list): {empty.sum():,} ({100 * empty.mean():.2f}%)  "
      f"=> 'predict all empty' baseline F0.5 = {empty.mean():.4f}")
print(f"total matched ids: {len(flat):,}   S2: {is2.sum():,}  S3: {is3.sum():,}  other: {(~is2 & ~is3).sum()}")
print(f"unique matched ids: {pc.count_distinct(flat).as_py():,}  (== total => each S2/S3 record belongs to <=1 S1)")
print("\nlist length distribution:")
vals, cnts = np.unique(lens, return_counts=True)
for v, c in zip(vals, cnts):
    print(f"  {v:>3}: {c:>9,} ({100 * c / n:5.2f}%)")
print(f"mean list length (non-singletons): {lens[lens > 0].mean():.3f}")
print("\n(#S2, #S3) joint distribution, top 20:")
pairs, pc_ = np.unique(np.stack([n2, n3], 1), axis=0, return_counts=True)
for i in np.argsort(-pc_)[:20]:
    print(f"  S2={pairs[i][0]}, S3={pairs[i][1]}: {pc_[i]:>9,} ({100 * pc_[i] / n:5.2f}%)")
print(f"  S1 with >=1 S2 match: {(n2 > 0).mean():.4f}   with >=1 S3 match: {(n3 > 0).mean():.4f}")
print(f"  S1 with S2 only: {((n2 > 0) & (n3 == 0)).mean():.4f}  S3 only: {((n2 == 0) & (n3 > 0)).mean():.4f}  both: {((n2 > 0) & (n3 > 0)).mean():.4f}")

# ---- coverage vs source files --------------------------------------------------------
s1 = read(f"{BASE}\\train\\train_source1.tsv", ["entity_id", "country"])
s1id, s1co = col(s1, "entity_id"), col(s1, "country")
pos1 = pc.index_in(g1, value_set=s1id).to_numpy(zero_copy_only=False)
print(f"\nGT S1 ids found in train_source1: {np.sum(pos1 >= 0):,} / {n:,}")
print(f"Spearman(GT row order, S1 file row order): {spearman(np.arange(n), pos1):.4f}")
co1 = np.array(s1co.to_pylist())[pos1]
for c in np.unique(co1):
    m = co1 == c
    print(f"  country={c}: n={m.sum():,} singleton rate={empty[m].mean():.4f}  mean len(non-single)={lens[m & ~empty].mean():.3f}")

res = {}
for src, mask in (("2", is2), ("3", is3)):
    s = read(f"{BASE}\\train\\train_source{src}.tsv", ["entity_id", "country"])
    sid, sco = col(s, "entity_id"), col(s, "country")
    ids = pc.filter(flat, pa.array(mask))
    pos = pc.index_in(ids, value_set=sid).to_numpy(zero_copy_only=False)
    par = parent[mask]
    print(f"\n--- Source {src}: file rows {len(sid):,}")
    print(f"  matched ids found in file: {np.sum(pos >= 0):,} / {len(ids):,}")
    print(f"  file records matched to some S1: {len(np.unique(pos[pos >= 0])):,} "
          f"({100 * len(np.unique(pos[pos >= 0])) / len(sid):.2f}%)  -> rest are unmatched distractors")
    cos = np.array(sco.to_pylist())
    print(f"  file country counts: {dict(zip(*np.unique(cos, return_counts=True)))}")
    same = cos[pos] == co1[par]
    print(f"  matched pair country agreement: {same.mean():.5f}  (disagree n={np.sum(~same):,})")
    if (~same).any():
        dis = list(zip(co1[par][~same][:10], cos[pos][~same][:10]))
        print(f"    sample disagreements (S1, S{src}): {dis}")
    # leakage: position / numeric-id correlation
    print(f"  Spearman(S1 file pos, S{src} file pos) over matched pairs: {spearman(pos1[par], pos):.4f}")
    num1 = np.array([int(x[3:]) for x in g1.take(pa.array(par)).to_pylist()], dtype=np.int64)
    numx = np.array([int(x[3:]) for x in ids.to_pylist()], dtype=np.int64)
    print(f"  Spearman(S1 numeric id, S{src} numeric id): {spearman(num1, numx):.4f}")
    # adjacency of same-cluster records within the file
    order = np.lexsort((pos, par))
    p_sorted, par_sorted = pos[order], par[order]
    same_cluster = par_sorted[1:] == par_sorted[:-1]
    gaps = np.abs(np.diff(p_sorted))[same_cluster]
    if len(gaps):
        print(f"  within-cluster |row gap| median={np.median(gaps):,.0f} "
              f"(random expectation ~{len(sid) / 3:,.0f}); frac gap<=5: {(gaps <= 5).mean():.4f}")
    res[src] = (sid, pos)
    del s, sco, cos
    gc.collect()

# ---- train/test overlap ----------------------------------------------------------------
print("\n--- train/test ID overlap")
for src in ("1", "2", "3"):
    tr = col(read(f"{BASE}\\train\\train_source{src}.tsv", ["entity_id"]), "entity_id")
    te = col(read(f"{BASE}\\test\\test_source{src}.tsv", ["entity_id"]), "entity_id")
    ov = pc.sum(pc.is_in(te, value_set=tr)).as_py()
    print(f"  source{src}: train {len(tr):,}  test {len(te):,}  test ids also in train: {ov:,}")
    del tr, te
    gc.collect()

# same (name,address) strings shared between train S1 and test S1?
a = read(f"{BASE}\\train\\train_source1.tsv", ["business_name", "business_address"])
b = read(f"{BASE}\\test\\test_source1.tsv", ["business_name", "business_address"])
ka = pc.binary_join_element_wise(col(a, "business_name"), col(a, "business_address"), "|")
kb = pc.binary_join_element_wise(col(b, "business_name"), col(b, "business_address"), "|")
print(f"  test S1 (name|address) exactly present in train S1: {pc.sum(pc.is_in(kb, value_set=ka)).as_py():,}")
na = pc.utf8_lower(col(a, "business_name"))
nb = pc.utf8_lower(col(b, "business_name"))
print(f"  test S1 lower(name) present in train S1 names: {pc.sum(pc.is_in(nb, value_set=na)).as_py():,} / {len(nb):,}")
