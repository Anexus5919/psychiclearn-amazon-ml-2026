"""Unit tests for the metric, normalisers, decision rule and writer (run: pytest tests/ from the package root).

L2 additions: tests for decide_v2 (threshold/expected_f05/prob_sum) and tune_v2.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ber import decide, normalize, scoring, writer  # noqa: E402


# ================================================================== original tests (unchanged)

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


# ================================================================== L2: decide_v2 tests

def _make_oof_df():
    """Small synthetic OOF DataFrame with src column for testing decide_v2."""
    # Two entities: A has 3 S2 candidates, B has 2 S3 + 1 S2 candidate
    return pd.DataFrame({
        "s1_id":   ["A", "A", "A", "A", "B", "B", "B"],
        "cand_id": ["c1","c2","c3","c4","c5","c6","c7"],
        "src":     [  2,   2,   2,   2,   3,   3,   2],
        "p":       [0.9, 0.8, 0.5, 0.2, 0.75, 0.4, 0.85],
        "label":   [  1,   1,   0,   0,   1,   0,   1],
        "country": ["US","US","US","US","IN","IN","IN"],
    })


class TestDecideV2Threshold:
    """Tests for decide_v2 with method='threshold'."""

    def test_basic_threshold_no_src(self):
        df = pd.DataFrame({
            "s1_id":   ["A", "A", "B", "B"],
            "cand_id": ["x", "y", "x", "z"],
            "p":       [0.9, 0.5, 0.6, 0.3],
        })
        params = {"method": "threshold", "t1_s2": 0.2, "t2_s2": 0.45}
        out = decide.decide_v2(df, params)
        # x (rank-1 for A, p=0.9>=0.2) and y (rank-2 for A, p=0.5>=0.45) both kept
        # x exclusive → goes to A; z (rank-1 for B, p=0.3>=0.2) kept
        assert "A" in out and set(out["A"]) == {"x", "y"}
        assert "B" in out and set(out["B"]) == {"z"}

    def test_per_source_thresholds(self):
        df = _make_oof_df()
        # Tight S3 threshold that rejects c6 (p=0.4 < 0.5)
        params = {
            "method": "threshold",
            "t1_s2": 0.10, "t2_s2": 0.10,
            "t1_s3": 0.50, "t2_s3": 0.50,
        }
        out = decide.decide_v2(df, params)
        # c6 (S3, p=0.4) should be rejected; c5 (S3, p=0.75) and c7 (S2, p=0.85) kept for B
        assert "B" in out
        assert "c6" not in out.get("B", [])
        assert "c5" in out.get("B", []) or "c7" in out.get("B", [])

    def test_t3_rank3_threshold(self):
        df = pd.DataFrame({
            "s1_id":   ["A", "A", "A"],
            "cand_id": ["c1","c2","c3"],
            "src":     [  2,   2,   2],
            "p":       [0.9, 0.8, 0.75],
        })
        # t2=0.70 would accept c3 (p=0.75), but t3=0.80 should reject it (rank=3)
        params = {"method": "threshold", "t1_s2": 0.3, "t2_s2": 0.70, "t3": 0.80}
        out = decide.decide_v2(df, params)
        assert "c3" not in out.get("A", [])
        assert "c1" in out.get("A", []) and "c2" in out.get("A", [])

    def test_exclusive_assignment(self):
        # c1 is contested between A and B; A scores it higher → goes to A only
        df = pd.DataFrame({
            "s1_id":   ["A",  "B"],
            "cand_id": ["c1", "c1"],
            "src":     [  2,    2],
            "p":       [0.9,  0.7],
        })
        params = {"method": "threshold", "t1_s2": 0.2, "t2_s2": 0.5}
        out = decide.decide_v2(df, params)
        assert "c1" in out.get("A", [])
        assert "c1" not in out.get("B", [])

    def test_singleton_entity_correctly_empty(self):
        # Entity A has no candidates that meet the threshold
        df = pd.DataFrame({
            "s1_id":   ["A"],
            "cand_id": ["c1"],
            "src":     [  2],
            "p":       [0.05],
        })
        params = {"method": "threshold", "t1_s2": 0.5, "t2_s2": 0.7}
        out = decide.decide_v2(df, params)
        assert "A" not in out  # empty prediction = correct for a singleton


class TestDecideV2ExpectedF05:
    """Tests for decide_v2 with method='expected_f05'."""

    def test_expected_f05_basic(self):
        # Entity with 2 high-p candidates: both should be picked
        df = pd.DataFrame({
            "s1_id":   ["A", "A", "A"],
            "cand_id": ["c1","c2","c3"],
            "src":     [  2,   2,   2],
            "p":       [0.9, 0.85, 0.05],
        })
        params = {"method": "expected_f05", "n_true_est": 2.0, "max_k": 10}
        out = decide.decide_v2(df, params)
        # c1 and c2 (high p) should be selected, c3 (p=0.05) probably not
        assert "A" in out
        assert "c1" in out["A"]
        assert "c2" in out["A"]

    def test_singleton_expected_f05(self):
        # n_true_est=0 → entity is a singleton → expect empty output
        df = pd.DataFrame({
            "s1_id":   ["A"],
            "cand_id": ["c1"],
            "src":     [  2],
            "p":       [0.8],
        })
        params = {"method": "expected_f05", "n_true_est": 0.0}
        out = decide.decide_v2(df, params)
        assert "A" not in out  # correct: empty prediction for a singleton

    def test_high_n_true_picks_more(self):
        # With n_true_est=5, expected-F0.5 should select more candidates
        df = pd.DataFrame({
            "s1_id":   ["A"] * 6,
            "cand_id": [f"c{i}" for i in range(6)],
            "src":     [2] * 6,
            "p":       [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        })
        params_hi = {"method": "expected_f05", "n_true_est": 5.0}
        params_lo = {"method": "expected_f05", "n_true_est": 1.0}
        out_hi = decide.decide_v2(df, params_hi)
        out_lo = decide.decide_v2(df, params_lo)
        assert len(out_hi.get("A", [])) >= len(out_lo.get("A", []))


class TestDecideV2ProbSum:
    """Tests for decide_v2 with method='prob_sum'."""

    def test_prob_sum_basic(self):
        # sum(p) = 0.9+0.8+0.1 = 1.8 → round → 2 candidates
        df = pd.DataFrame({
            "s1_id":   ["A", "A", "A"],
            "cand_id": ["c1","c2","c3"],
            "src":     [  2,   2,   2],
            "p":       [0.9, 0.8, 0.1],
        })
        params = {"method": "prob_sum", "ps_t1": 0.02, "ps_tmin": 0.05, "ps_cap": 10}
        out = decide.decide_v2(df, params)
        assert "A" in out
        assert len(out["A"]) == 2  # round(1.8) = 2

    def test_prob_sum_rank1_gate(self):
        # Rank-1 p below ps_t1 → empty prediction
        df = pd.DataFrame({
            "s1_id":   ["A"],
            "cand_id": ["c1"],
            "src":     [  2],
            "p":       [0.01],
        })
        params = {"method": "prob_sum", "ps_t1": 0.05, "ps_tmin": 0.05}
        out = decide.decide_v2(df, params)
        assert "A" not in out

    def test_prob_sum_cap(self):
        # Even if sum(p) is large, ps_cap limits selections
        df = pd.DataFrame({
            "s1_id":   ["A"] * 10,
            "cand_id": [f"c{i}" for i in range(10)],
            "src":     [2] * 10,
            "p":       [0.9] * 10,  # sum=9, round=9, but cap=3
        })
        params = {"method": "prob_sum", "ps_t1": 0.02, "ps_tmin": 0.05, "ps_cap": 3}
        out = decide.decide_v2(df, params)
        assert len(out.get("A", [])) <= 3


class TestTuneV2:
    """Tests for tune_v2: half-business split, no leakage, returns params."""

    def _make_df(self, seed=42):
        """Build a synthetic OOF dataframe for 20 entities."""
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(20):
            s1 = f"S1-{i:04d}"
            n_cand = rng.integers(3, 8)
            n_true = rng.integers(1, 4)
            probs = sorted(rng.uniform(0.1, 0.95, n_cand), reverse=True)
            for j, p in enumerate(probs):
                label = 1 if j < n_true else 0
                src = 2 if j % 2 == 0 else 3
                rows.append({"s1_id": s1, "cand_id": f"C{i}-{j}", "src": src,
                             "p": float(p), "label": label, "country": "US"})
        return pd.DataFrame(rows)

    def _make_truth(self, df):
        return (df[df["label"] == 1]
                .groupby("s1_id")["cand_id"]
                .apply(set)
                .to_dict())

    def test_tune_v2_threshold_returns_valid_params(self):
        df = self._make_df()
        truth = self._make_truth(df)
        tune_f, held_f, params = decide.tune_v2(
            df, truth, method="threshold",
            grid1=np.array([0.2, 0.5]),
            grid2=np.array([0.5, 0.7]),
            grid3=np.array([999.0]),  # disable t3
        )
        assert 0.0 <= tune_f <= 1.0
        assert 0.0 <= held_f <= 1.0
        assert params["method"] == "threshold"
        assert "t1_s2" in params and "t2_s2" in params

    def test_tune_v2_expected_f05_returns_valid_params(self):
        df = self._make_df()
        truth = self._make_truth(df)
        tune_f, held_f, params = decide.tune_v2(df, truth, method="expected_f05")
        assert 0.0 <= tune_f <= 1.0
        assert 0.0 <= held_f <= 1.0
        assert params["method"] == "expected_f05"

    def test_tune_v2_prob_sum_returns_valid_params(self):
        df = self._make_df()
        truth = self._make_truth(df)
        tune_f, held_f, params = decide.tune_v2(
            df, truth, method="prob_sum",
            grid1=np.array([0.1, 0.2]),
            grid2=np.array([0.1, 0.3]),
        )
        assert 0.0 <= tune_f <= 1.0
        assert 0.0 <= held_f <= 1.0
        assert params["method"] == "prob_sum"

    def test_tune_v2_no_leakage(self):
        """Tune and held halves are completely disjoint."""
        df = self._make_df()
        truth = self._make_truth(df)

        tune_ids = set()
        held_ids = set()

        def half_selector_spy(s1_id):
            result = hash(s1_id) % 2 == 0
            if result:
                tune_ids.add(s1_id)
            else:
                held_ids.add(s1_id)
            return result

        decide.tune_v2(df, truth, method="threshold",
                       half_selector=half_selector_spy,
                       grid1=np.array([0.2, 0.5]),
                       grid2=np.array([0.5, 0.7]),
                       grid3=np.array([999.0]))

        assert len(tune_ids & held_ids) == 0, "Tune and held halves must be disjoint"

    def test_tune_v2_custom_half_selector(self):
        """Custom half_selector is respected."""
        df = self._make_df()
        truth = self._make_truth(df)
        all_ids = sorted(truth.keys())
        tune_set = set(all_ids[:10])

        def my_selector(s1_id):
            return s1_id in tune_set

        tune_f, held_f, params = decide.tune_v2(
            df, truth, method="prob_sum",
            half_selector=my_selector,
            grid1=np.array([0.1, 0.2]),
            grid2=np.array([0.1, 0.3]),
        )
        assert 0.0 <= tune_f <= 1.0
        assert 0.0 <= held_f <= 1.0

    def test_score_v2_matches_scoring_module(self):
        """_score_v2 should agree with scoring.macro_f05 for the threshold method."""
        df = self._make_df()
        truth = self._make_truth(df)
        params = {"method": "threshold", "t1_s2": 0.5, "t2_s2": 0.7}
        score_v2 = decide._score_v2(df, truth, params)
        pred_map = decide.decide_v2(df, params)
        score_ref = scoring.macro_f05(
            {k: list(v) for k, v in pred_map.items()},
            truth,
        )
        assert abs(score_v2 - score_ref) < 1e-9


class TestDecideV2Idempotency:
    """decide_v2 must be side-effect free: calling twice returns the same result."""

    def test_idempotent(self):
        df = _make_oof_df()
        params = {"method": "threshold", "t1_s2": 0.5, "t2_s2": 0.7}
        out1 = decide.decide_v2(df.copy(), params)
        out2 = decide.decide_v2(df.copy(), params)
        assert out1 == out2

    def test_original_df_not_mutated(self):
        df = _make_oof_df()
        orig_cols = set(df.columns)
        _ = decide.decide_v2(df, {"method": "threshold", "t1_s2": 0.5, "t2_s2": 0.7})
        # decide_v2 must not add columns to the caller's dataframe
        assert set(df.columns) == orig_cols
