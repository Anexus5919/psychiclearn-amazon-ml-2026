"""Writing and self-checking the two submission files.

Files are written with LF line endings: the official validator splits rows on "\\n" only, so a
Windows "\\r\\n" would silently glue "\\r" to the last ID of every row.
"""
import os


def write_id_lists(path, second_col, s1_ids_in_order, mapping):
    """Write one row per S1 id (in the given order), comma-joined IDs, LF endings.

    mapping values are lists of IDs or of already comma-joined ID chunks; duplicates are removed.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{second_col}\n")
        for s1 in s1_ids_in_order:
            ids = ",".join(mapping.get(s1, ())).split(",") if s1 in mapping else []
            ids = [i for i in dict.fromkeys(ids) if i]
            f.write(s1 + "\t" + ",".join(ids) + "\n")


def _rows(path, col):
    with open(path, "rb") as fb:
        head = fb.read(1 << 20)
    assert b"\r" not in head, f"{path}: contains CR characters"
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline()
        assert header == f"source1_entity_id\t{col}\n", f"{path}: bad header {header!r}"
        for ln in f:
            assert ln.endswith("\n") and "\r" not in ln, f"{path}: bad line ending"
            s1, _, rest = ln[:-1].partition("\t")
            yield s1, (rest.split(",") if rest else [])


def check_outputs(matching_path, candidate_path, s1_ids, valid_targets):
    """Stream both files and assert every rule we can check locally; returns a small stats dict."""
    required = set(s1_ids)
    stats, matches = {}, {}
    for path, col in ((matching_path, "matched_entity_ids"), (candidate_path, "candidate_entity_ids")):
        seen, empty, total = set(), 0, 0
        for s1, ids in _rows(path, col):
            assert s1 not in seen, f"{path}: duplicate row {s1}"
            seen.add(s1)
            assert len(ids) == len(set(ids)), f"{path}: duplicate id in list for {s1}"
            for i in ids:
                assert i.startswith(("S2-", "S3-")), f"{path}: bad prefix {i}"
                assert i in valid_targets, f"{path}: unknown id {i}"
            if col == "matched_entity_ids":
                if ids:
                    matches[s1] = set(ids)
            elif s1 in matches:
                assert matches[s1] <= set(ids), f"{s1}: matches outside candidates"
            empty += not ids
            total += len(ids)
        assert seen == required, f"{path}: S1 rows differ from test S1 ({len(seen)} vs {len(required)})"
        stats[col] = {"rows": len(seen), "empty": empty, "mean_len": total / max(1, len(seen))}
    return stats
