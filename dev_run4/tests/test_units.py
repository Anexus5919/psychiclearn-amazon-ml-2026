"""Unit tests for the metric, normalisers, decision rule and writer (run: pytest tests/ from the package root)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber import decide, normalize, scoring, writer  # noqa: E402


def test_metric_matches_problem_statement_example():
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    assert abs(scoring.f05_entity(pred, true) - 0.7142857) < 1e-6
    assert scoring.f05_entity(set(), set()) == 1.0
    assert scoring.f05_entity({"S2-1"}, set()) == 0.0
    assert scoring.f05_entity(set(), {"S2-1"}) == 0.0


def test_name_normalisation_real_noise():
    v = normalize.name_views("Jaxevo trading as Mansfield Devices", "US")
    assert v["name_core"] == "mansfield devices" and v["name_dba"]
    assert normalize.name_views("Esparza Ridge Sunrise Ínc", "US")["name_core"] == "esparza ridge sunrise"
    assert normalize.name_views("@bachmannsbistro", "US")["name_handle"]
    assert normalize.name_views("capitalholding.com", "US")["name_core"] == "capitalholding"
    assert normalize.name_views("UrologypartnersCom", "US")["name_core"] == "urologypartners"
    assert normalize.name_views("Dss Care L.L.P.", "India")["legal"] == "llp"
    assert normalize.name_views("Gagny (France) Club (S.A.R.L.)", "France")["name_core"] == "gagny club"
    v = normalize.name_views("कृष्ण फाइनेंस प्राइवेट लिमिटेड", "India")
    assert v["name_native"] and "फाइनेंस" in v["name_core"]  # Indic vowel signs are preserved
    assert normalize.name_views("Gallagher Crystal John LLC", "US")["name_acr"] == "gcj"


def test_address_normalisation_real_noise():
    a = normalize.addr_views("108 2ST ST, PMB 3330, ROME, OH", "US")
    b = normalize.addr_views("OH, 108 2nd Street, Rome", "US")
    assert set(a["addr_core"].split()) == set(b["addr_core"].split())
    assert a["addr_nums"] == "108 2"
    assert normalize.addr_views("008644 ASHLEY GLEN CIRCLE", "US")["addr_primary"] == "8644"
    assert normalize.addr_views("", "US")["addr_empty"]
    assert normalize.addr_views("1130 RIVER ROCK COURT, NULL, KIEL, WI", "US")["addr_core"] == "1130 river rock ct kiel wi"
    fr = normalize.addr_views("N° 50 R. DE LA BENAUGE, BORDEAUX", "France")
    assert "rue" in fr["addr_core"].split() and fr["addr_primary"] == "50"


def test_decide_exclusive_and_rank_thresholds():
    df = pd.DataFrame({"s1_id": ["A", "A", "B", "B"], "cand_id": ["x", "y", "x", "z"], "p": [0.9, 0.5, 0.6, 0.3]})
    out = decide.decide(df, t1=0.2, t2=0.45)
    assert out == {"A": ["x", "y"], "B": ["z"]}  # x goes to A only; z is B's rank-1 after assignment
    df["label"] = [1, 1, 0, 1]
    f, t1, t2 = decide.tune_thresholds(df, {"A": {"x", "y"}, "B": {"z"}}, grid1=np.array([0.2]), grid2=np.array([0.45]))
    assert abs(f - 1.0) < 1e-9


def test_writer_lf_and_checks(tmp_path):
    m, c = tmp_path / "m.tsv", tmp_path / "c.tsv"
    writer.write_id_lists(str(c), "candidate_entity_ids", ["S1-1", "S1-2"], {"S1-1": ["S2-1", "S3-2", "S2-1"]})
    writer.write_id_lists(str(m), "matched_entity_ids", ["S1-1", "S1-2"], {"S1-1": ["S2-1"]})
    assert b"\r" not in m.read_bytes()
    stats = writer.check_outputs(str(m), str(c), ["S1-1", "S1-2"], {"S2-1", "S3-2"})
    assert stats["matched_entity_ids"]["rows"] == 2 and stats["matched_entity_ids"]["empty"] == 1
