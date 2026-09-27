import io, os, py_compile
p = r"C:\Users\ANEXUS\Downloads\PsychicLearn_dev\src\ber\pipeline.py"
s = io.open(p, encoding="utf-8").read()
rep = [
('''    k_region: int = 0               # >0 enables the region-restricted name pass (see ber.regions)
''',
'''    k_region: int = 0               # >0 enables the region-restricted name pass (see ber.regions)
    cand_min_p: float = 0.0001      # candidate_pairs.tsv keeps pairs with matcher p >= this (run-4 OOF:
                                    # 22.6 -> 7.3 candidates per S1 for a 0.003% candidate-recall loss)
'''),
('''        for s1, joined in keys.groupby("s1_id", sort=False)["cand_id"].agg(",".join).items():''',
'''        kc = keys[keys["p"] >= cfg.cand_min_p]
        for s1, joined in kc.groupby("s1_id", sort=False)["cand_id"].agg(",".join).items():'''),
('''                      ("k_reverse", int), ("exact_name_cap", int), ("k_region", int),''',
'''                      ("k_reverse", int), ("exact_name_cap", int), ("k_region", int), ("cand_min_p", float),'''),
('''    for k in ("n_jobs", "train_frac", "k_name", "k_addr", "k_combo", "k_reverse", "exact_name_cap", "k_region",''',
'''    for k in ("n_jobs", "train_frac", "k_name", "k_addr", "k_combo", "k_reverse", "exact_name_cap", "k_region", "cand_min_p",'''),
]
for a, b in rep:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
io.open(p + ".new", "w", encoding="utf-8", newline="\n").write(s)
py_compile.compile(p + ".new", doraise=True)
os.replace(p + ".new", p)
print("ok")
