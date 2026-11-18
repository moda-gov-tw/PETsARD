import logging
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from anonymeter.evaluators import SinglingOutEvaluator

from petsard.evaluator.anonymeter import Anonymeter
from petsard.exceptions import UnsupportedMethodError
from petsard.utils import safe_round


class TestAnonymeter(unittest.TestCase):
    """Test for Anonymeter Evaluator."""

    def setUp(self):
        """Set up test fixtures."""
        # Test configuration
        self.config = {
            "eval_method": "anonymeter-singlingout",
            "n_attacks": 100,
            "n_cols": 2,
            "max_attempts": 1000,
            "mode": "multivariate",
        }

        # Create mock data with more rows to avoid sampling issues
        import numpy as np

        np.random.seed(42)  # For reproducible results

        n_rows = 200  # Enough rows for sampling
        self.data = {
            "ori": pd.DataFrame(
                {
                    "col1": np.random.randint(1, 100, n_rows),
                    "col2": np.random.choice(["a", "b", "c", "d", "e"], n_rows),
                    "col3": np.random.uniform(1.0, 10.0, n_rows),
                }
            ),
            "syn": pd.DataFrame(
                {
                    "col1": np.random.randint(1, 100, n_rows),
                    "col2": np.random.choice(["a", "b", "c", "d", "e"], n_rows),
                    "col3": np.random.uniform(1.0, 10.0, n_rows),
                }
            ),
            "control": pd.DataFrame(
                {
                    "col1": np.random.randint(100, 200, n_rows),
                    "col2": np.random.choice(["f", "g", "h", "i", "j"], n_rows),
                    "col3": np.random.uniform(10.0, 20.0, n_rows),
                }
            ),
        }

    @patch("anonymeter.evaluators.SinglingOutEvaluator")
    def test_init(self, mock_evaluator):
        """Test initialization."""
        evaluator = Anonymeter(config=self.config)
        self.assertEqual(evaluator.config, self.config)
        self.assertIsNone(evaluator._impl)

    def test_eval_singlingout(self):
        """Test eval method with SinglingOut."""
        # Use a smaller n_attacks to avoid sampling issues
        small_config = self.config.copy()
        small_config["n_attacks"] = 10
        small_config["max_attempts"] = 100

        # Execute evaluator
        evaluator = Anonymeter(config=small_config)
        result = evaluator.eval(self.data)

        # Assert results structure
        self.assertIn("global", result)
        self.assertIn("details", result)

        # Assert global results content
        global_data = result["global"]
        self.assertIsInstance(global_data, pd.DataFrame)
        result_dict = global_data.iloc[0].to_dict()
        self.assertIn("risk", result_dict)
        self.assertIn("risk_CI_btm", result_dict)
        self.assertIn("risk_CI_top", result_dict)
        self.assertIn("attack_rate", result_dict)

        # Check that risk is a valid number (not NaN)
        self.assertIsNotNone(result_dict["risk"])
        self.assertFalse(pd.isna(result_dict["risk"]))

    def test_eval_linkability(self):
        """Test eval method with Linkability."""
        # Setup config for linkability with smaller n_attacks
        linkability_config = {
            "eval_method": "anonymeter-linkability",
            "n_attacks": 10,  # Reduced to avoid sampling issues
            "n_neighbors": 5,
            "aux_cols": [["col1"], ["col3"]],
            "n_jobs": -1,
        }

        # Execute evaluator
        evaluator = Anonymeter(config=linkability_config)
        result = evaluator.eval(self.data)

        # Assert structure and content
        self.assertIn("global", result)
        self.assertIsInstance(result["global"], pd.DataFrame)

        # Check that risk is a valid number (not NaN)
        risk_value = result["global"].iloc[0]["risk"]
        self.assertIsNotNone(risk_value)
        self.assertFalse(pd.isna(risk_value))

    def test_invalid_method(self):
        """Test with invalid evaluation method."""
        with self.assertRaises(UnsupportedMethodError):
            Anonymeter(config={"eval_method": "anonymeter-invalid"})


if __name__ == "__main__":
    unittest.main()


# ---- SinglingOut with column names that pandas queries cannot use as they are ----
# 欄位名稱含「-」等字元時之 SinglingOut 測試，使用公開範例資料 adult-income

ADULT_CSV = (
    Path(__file__).resolve().parents[2]
    / "tutorial"
    / "demo_csv"
    / "data"
    / "adult_income.csv"
)
SINGLINGOUT_CONFIG = {
    "eval_method": "anonymeter-singlingout",
    "n_attacks": 50,
    "n_cols": 3,
    "max_attempts": 600,
}


@pytest.fixture(scope="module")
def adult_splits():
    if not ADULT_CSV.exists():
        pytest.skip("needs tutorial/demo_csv/data/adult_income.csv")
    df = pd.read_csv(ADULT_CSV, nrows=2500)
    return {
        "ori": df.iloc[:1000].reset_index(drop=True),
        "syn": df.iloc[1000:2000].reset_index(drop=True),
        "control": df.iloc[2000:2500].reset_index(drop=True),
    }


def _seed_anonymeter(monkeypatch, seed: int = 0) -> None:
    """Anonymeter draws its queries from a module-level generator; fix it."""
    from anonymeter.evaluators import singling_out_evaluator

    monkeypatch.setattr(singling_out_evaluator, "rng", np.random.default_rng(seed))


def _direct_anonymeter(data: dict) -> SinglingOutEvaluator:
    """What the evaluator did before the name mapping: the data go to Anonymeter unchanged."""
    evaluator = SinglingOutEvaluator(
        ori=data["ori"],
        syn=data["syn"],
        control=data["control"],
        n_attacks=SINGLINGOUT_CONFIG["n_attacks"],
        n_cols=SINGLINGOUT_CONFIG["n_cols"],
        max_attempts=SINGLINGOUT_CONFIG["max_attempts"],
    )
    return evaluator.evaluate(mode="multivariate")


def _failed_query_logs(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if "failed queries" in r.getMessage()
    ]


class TestSinglingOutColumnNames:
    # 含「-」欄位：修改後查詢不再失敗，有效查詢數增加；輸出之查詢改回原欄名，且可在原始資料上執行
    def test_names_with_dashes_no_longer_make_queries_fail(
        self, adult_splits, monkeypatch, caplog
    ):
        assert {"educational-num", "marital-status", "capital-gain"} <= set(
            adult_splits["ori"].columns
        )
        caplog.set_level(logging.WARNING)

        _seed_anonymeter(monkeypatch)
        before = _direct_anonymeter(adult_splits)
        failed_before = _failed_query_logs(caplog)
        caplog.clear()

        _seed_anonymeter(monkeypatch)
        evaluator = Anonymeter(config=dict(SINGLINGOUT_CONFIG))
        result = evaluator.eval(adult_splits)
        failed_after = _failed_query_logs(caplog)

        assert failed_before  # the unchanged names make queries fail
        assert not failed_after
        assert evaluator._column_mapping["marital-status"] == "marital_status"
        attack_queries = result["details"]["attack_queries"]
        assert len(attack_queries) > len(before.queries())

        # the queries name the original columns and single out one record of the original data
        ori = adult_splits["ori"].drop_duplicates()
        assert any("`marital-status`" in q for q in attack_queries)
        assert not any("marital_status" in q for q in attack_queries)
        for query in attack_queries:
            assert len(ori.query(query, engine="python")) == 1

    # 欄位名稱無特殊字元：與修改前(資料原樣交給 Anonymeter)結果完全相同
    def test_safe_names_give_the_same_result_as_before(self, adult_splits, monkeypatch):
        safe = {
            key: df.rename(columns=lambda c: c.replace("-", "_"))
            for key, df in adult_splits.items()
        }

        _seed_anonymeter(monkeypatch)
        before = _direct_anonymeter(safe)

        _seed_anonymeter(monkeypatch)
        evaluator = Anonymeter(config=dict(SINGLINGOUT_CONFIG))
        result = evaluator.eval(safe)

        assert evaluator._column_mapping == {}
        assert result["details"]["attack_queries"] == before.queries()
        assert result["details"]["baseline_queries"] == before.queries(baseline=True)
        scores = result["global"].iloc[0]
        risk = before.risk()
        assert scores["risk"] == safe_round(risk.value)
        assert scores["risk_CI_btm"] == safe_round(risk.ci[0])
        assert scores["risk_CI_top"] == safe_round(risk.ci[1])
        rates = before.results()
        for rate_type in ["attack_rate", "baseline_rate", "control_rate"]:
            assert scores[rate_type] == safe_round(getattr(rates, rate_type).value)
            assert scores[f"{rate_type}_err"] == safe_round(
                getattr(rates, rate_type).error
            )

    # 非字串欄名(例如整數 0、1)：DataFrame.query 無法引用，查詢保留安全名稱；以名稱對應改名後可在原始資料上執行
    def test_non_string_names_keep_the_safe_names(self, adult_splits, monkeypatch):
        numbered = {
            key: df.set_axis(range(df.shape[1]), axis=1)
            for key, df in adult_splits.items()
        }

        _seed_anonymeter(monkeypatch)
        evaluator = Anonymeter(config=dict(SINGLINGOUT_CONFIG))
        result = evaluator.eval(numbered)

        mapping = evaluator._column_mapping
        assert mapping == {i: f"_{i}" for i in range(adult_splits["ori"].shape[1])}
        attack_queries = result["details"]["attack_queries"]
        assert attack_queries
        assert not any("`" in q for q in attack_queries)

        renamed = {key: df.rename(columns=mapping) for key, df in numbered.items()}
        _seed_anonymeter(monkeypatch)
        before = _direct_anonymeter(renamed)
        assert attack_queries == before.queries()
        ori = renamed["ori"].drop_duplicates()
        for query in attack_queries:
            assert len(ori.query(query, engine="python")) == 1

    # 名稱對應規則：只改不安全的名稱、避免重名，查詢改回原名時不動引號內的值
    def test_query_safe_names_and_restoring_them(self):
        from petsard.evaluator.anonymeter import _query_safe_names, _restore_query_names

        assert _query_safe_names(["age", "income", "年齡"]) == {}
        mapping = _query_safe_names(
            ["educational-num", "educational_num", "capital gain", "1st", "class"]
        )
        assert mapping == {
            "educational-num": "educational_num_2",
            "capital gain": "capital_gain",
            "1st": "_1st",
            "class": "_class",
        }
        reverse = {safe: original for original, safe in mapping.items()}
        query = "educational_num_2 >= 5 & capital_gain == 'capital_gain' & _1st.isna()"
        assert (
            _restore_query_names(query, reverse)
            == "`educational-num` >= 5 & `capital gain` == 'capital_gain' & `1st`.isna()"
        )
