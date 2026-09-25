"""Reading the challenge TSVs and small helpers for Parquet artefacts."""
import os

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def _resolve(path):
    """Return `path` if it exists, else `path + '.gz'` (the S3 copy of the data is gzipped)."""
    if os.path.exists(path):
        return path
    if os.path.exists(path + ".gz"):
        return path + ".gz"
    raise FileNotFoundError(path)


def read_tsv(path, cols):
    """Read a challenge TSV with every column as a plain string.

    Uses an explicit tab delimiter and keeps literal values such as "NA"/"null" as text
    (no NA inference), and keeps the standard CSV quote handling, which correctly decodes the
    escaped quotes present in the test files. Accepts `.tsv` or `.tsv.gz`.
    """
    tb = pv.read_csv(
        _resolve(path),
        parse_options=pv.ParseOptions(delimiter="\t"),
        convert_options=pv.ConvertOptions(
            column_types={c: pa.string() for c in cols},
            strings_can_be_null=False,
            null_values=[],
            include_columns=cols,
        ),
        read_options=pv.ReadOptions(block_size=1 << 26),
    )
    return tb.to_pandas()


def read_source(data_dir, split, src):
    """Load `<split>/<split>_source<src>.tsv` and check the documented schema."""
    df = read_tsv(os.path.join(data_dir, split, f"{split}_source{src}.tsv"), SOURCE_COLS)
    assert df["entity_id"].is_unique, f"duplicate entity_id in {split} source{src}"
    assert df["entity_id"].str.startswith(f"S{src}-").all(), f"bad prefix in {split} source{src}"
    return df


def read_ground_truth(data_dir):
    """Return {source1_entity_id: set(matched ids)} from train_ground_truth.tsv."""
    df = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"),
                  ["source1_entity_id", "matched_entity_ids"])
    return {s1: {x for x in m.split(",") if x} for s1, m in zip(df["source1_entity_id"], df["matched_entity_ids"])}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
