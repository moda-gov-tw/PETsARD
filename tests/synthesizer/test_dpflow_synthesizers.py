"""Tests for the differentially private tabular synthesizers DPflow_Copula and DPflow_AdvGAN.

差分隱私表格合成器(DPflow_Copula、DPflow_AdvGAN)的測試，只使用測試內產生的小型假資料。
"""

import importlib.util
import json
import math

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.optimize import brentq

from petsard.exceptions import ConfigError, UnsupportedMethodError
from petsard.synthesizer import Synthesizer
from petsard.synthesizer.DPflow_common import eps_for_mu, mu_for, solve_noise_multiplier
from petsard.synthesizer.DPflow_Copula import DPflowCopulaSynthesizer
from petsard.synthesizer.synthesizer import SynthesizerMap

SPEC = [
    {"name": "age", "type": "num", "integer": True},
    {"name": "score", "type": "num"},
    {"name": "group", "type": "cat", "values": ["a", "b", "c", "?"]},
    {"name": "flag", "type": "cat", "values": [0, 1]},
]
HAS_ADVGAN = all(importlib.util.find_spec(m) is not None for m in ("torch", "opacus", "dp_accounting"))


def make_data(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, n),
            "score": rng.normal(50, 10, n),
            "group": rng.choice(["a", "b", "c", "?"], n),
            "flag": rng.integers(0, 2, n),
        }
    )


def make(method: str, **cfg):
    config = {"syn_method": method, "column_spec": SPEC, "epsilon": 4, "sample_num_rows": 200, **cfg}
    if method == "DPflow_AdvGAN":
        from petsard.synthesizer.DPflow_AdvGAN import DPflowAdvGANSynthesizer

        return DPflowAdvGANSynthesizer(config)
    return DPflowCopulaSynthesizer(config)


def run(method: str, data: pd.DataFrame, **cfg):
    syn = make(method, **cfg)
    syn.fit(data)
    return syn, syn.sample()


class TestMethodNames:
    def test_method_names_map_to_their_codes(self):
        assert SynthesizerMap.map("dpflow_copula") == SynthesizerMap.DPFLOW_COPULA
        assert SynthesizerMap.map("dpflow_advgan") == SynthesizerMap.DPFLOW_ADVGAN

    def test_synthesizer_front_end_runs_dpflow_copula(self):
        syn = Synthesizer(method="DPflow_Copula", sample_num_rows=50, column_spec=SPEC, epsilon=2, seed=0)
        syn.create()
        out = syn.fit_sample(make_data())
        assert len(out) == 50
        assert list(out.columns) == [c["name"] for c in SPEC]

    def test_unknown_method(self):
        with pytest.raises(UnsupportedMethodError):
            Synthesizer(method="DPflow_Unknown")


class TestDPflowCopula:
    def test_output_follows_column_spec(self):
        syn, out = run("DPflow_Copula", make_data(), seed=1)
        assert list(out.columns) == [c["name"] for c in SPEC]
        assert len(out) == 200
        assert pd.api.types.is_integer_dtype(out["age"])
        assert set(out["group"]) <= {"a", "b", "c", "?"}
        assert set(out["flag"]) <= {0, 1}
        assert syn.privacy["achieved_epsilon"] <= 4 + 1e-9
        assert syn.privacy["method"] == "DPflow_Copula"
        assert syn.privacy["n"] == 400

    def test_seed_reproduces_and_no_seed_uses_system_randomness(self):
        data = make_data()
        _, a = run("DPflow_Copula", data, seed=3)
        _, b = run("DPflow_Copula", data, seed=3)
        pd.testing.assert_frame_equal(a, b)
        s1, c = run("DPflow_Copula", data)
        _, d = run("DPflow_Copula", data)
        assert not c.equals(d)
        assert "seed" not in s1.privacy
        assert "operating-system" in s1.privacy["seed_source"]

    def test_privacy_report_file(self, tmp_path):
        path = tmp_path / "report.json"
        run("DPflow_Copula", make_data(), seed=0, privacy_report=str(path))
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["target_epsilon"] == 4
        assert report["delta"] == pytest.approx(1e-5)

    def test_column_spec_from_yaml_file(self, tmp_path):
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.safe_dump({"columns": SPEC}), encoding="utf-8")
        _, out = run("DPflow_Copula", make_data(), seed=0, column_spec=str(path))
        assert list(out.columns) == [c["name"] for c in SPEC]

    def test_missing_values_are_rejected(self):
        data = make_data()
        data.loc[0, "score"] = np.nan
        with pytest.raises(ConfigError, match="missing values"):
            run("DPflow_Copula", data)

    def test_value_outside_declared_list_is_rejected(self):
        data = make_data()
        data.loc[0, "group"] = "d"
        with pytest.raises(ConfigError, match="outside"):
            run("DPflow_Copula", data)

    def test_columns_must_match_spec(self):
        data = make_data().assign(extra=1)
        with pytest.raises(ConfigError, match="differ"):
            run("DPflow_Copula", data)

    # 未設定(或為 0)sample_num_rows：產生與訓練資料相同筆數
    def test_without_sample_num_rows_generates_training_rows(self):
        _, out = run("DPflow_Copula", make_data(150), seed=0, sample_num_rows=0)
        assert len(out) == 150
        syn = DPflowCopulaSynthesizer({"syn_method": "DPflow_Copula", "column_spec": SPEC, "epsilon": 2, "seed": 0})
        syn.fit(make_data(120))
        assert len(syn.sample()) == 120

    # 負的 sample_num_rows：與 DPflow_AdvGAN 相同，在花用隱私預算前以 ConfigError 拒絕，不產生隱私報告
    def test_negative_sample_num_rows_is_rejected_before_the_budget_is_spent(self, tmp_path):
        with pytest.raises(ConfigError, match="sample_num_rows"):
            make("DPflow_Copula", sample_num_rows=-5)
        with pytest.raises(ConfigError, match="sample_num_rows"):
            Synthesizer(method="DPflow_Copula", sample_num_rows=-5, column_spec=SPEC, epsilon=2, seed=0).create()
        report = tmp_path / "privacy.json"
        syn = make("DPflow_Copula", seed=0, privacy_report=str(report))
        syn.update_config({"sample_num_rows": -1})
        with pytest.raises(ConfigError, match="sample_num_rows"):
            syn.fit(make_data())
        assert syn.privacy == {}
        assert not report.exists()

    def test_required_parameters(self):
        with pytest.raises(ConfigError, match="epsilon"):
            DPflowCopulaSynthesizer({"syn_method": "DPflow_Copula", "column_spec": SPEC})
        with pytest.raises(ConfigError, match="column_spec"):
            DPflowCopulaSynthesizer({"syn_method": "DPflow_Copula", "epsilon": 1})
        with pytest.raises(ConfigError, match="values"):
            DPflowCopulaSynthesizer({"syn_method": "DPflow_Copula", "epsilon": 1,
                                     "column_spec": [{"name": "x", "type": "cat"}]})


class TestAccounting:
    def test_gaussian_dp_round_trip(self):
        for eps in (1, 4, 8, 16):
            assert eps_for_mu(mu_for(eps)) == pytest.approx(eps, rel=1e-9)

    @pytest.mark.parametrize("eps", [0.9, 1.0, 2.5, 8.0])
    def test_noise_multiplier_never_above_target(self, eps):
        def account(s):  # decreasing in s, like a DP-SGD accountant
            return 10 / s + 0.37 / s**2

        old = brentq(lambda s: account(s) - eps, 0.3, 200, xtol=1e-3)
        new = solve_noise_multiplier(account, eps, 0.3, 200)
        assert account(new) <= eps
        if account(old) <= eps:
            assert new == old  # a solution that already meets the target is kept unchanged
        else:
            assert old < new < old + 2e-3


@pytest.mark.skipif(not HAS_ADVGAN, reason="needs the advgan extra (torch, opacus, dp-accounting)")
class TestDPflowAdvGAN:
    def test_fit_sample_and_privacy_report(self):
        syn, out = run("DPflow_AdvGAN", make_data(), seed=0, epochs=1, batch_size=64, keep_from_seed=["flag"])
        assert list(out.columns) == [c["name"] for c in SPEC]
        assert len(out) == 200
        assert set(out["group"]) <= {"a", "b", "c", "?"}
        assert pd.api.types.is_integer_dtype(out["age"])
        assert syn.privacy["method"] == "DPflow_AdvGAN"
        assert syn.privacy["achieved_epsilon"] <= 4
        assert syn.privacy["noise_multiplier"] > 0
        assert math.isclose(syn.privacy["share_dp_sgd"], 0.3)

    def test_synthesizer_front_end_accepts_dpflow_advgan(self):
        syn = Synthesizer(method="DPflow_AdvGAN", sample_num_rows=30, column_spec=SPEC, epsilon=2, seed=0,
                          epochs=1, batch_size=64)
        syn.create()
        out = syn.fit_sample(make_data())
        assert len(out) == 30

    # 未設定 sample_num_rows：經 Synthesizer 前端 create/fit/sample，產生與訓練資料相同筆數
    def test_front_end_without_sample_num_rows_generates_training_rows(self):
        syn = Synthesizer(method="DPflow_AdvGAN", column_spec=SPEC, epsilon=2, seed=0, epochs=1, batch_size=64)
        syn.create()
        syn.fit(make_data(150))
        out = syn.sample()
        assert out.shape == (150, len(SPEC))
        assert syn._impl.privacy["achieved_epsilon"] <= 2

    # 直接使用且設定中沒有 sample_num_rows：同樣產生與訓練資料相同筆數(與 DPflow_Copula 一致)
    def test_direct_use_without_sample_num_rows_generates_training_rows(self):
        from petsard.synthesizer.DPflow_AdvGAN import DPflowAdvGANSynthesizer

        syn = DPflowAdvGANSynthesizer(
            {"syn_method": "DPflow_AdvGAN", "column_spec": SPEC, "epsilon": 2, "seed": 0, "epochs": 1})
        syn.fit(make_data(120))
        assert len(syn.sample()) == 120

    # 不合法的 sample_num_rows 在花用隱私預算前就被拒絕，不產生隱私報告
    def test_invalid_sample_num_rows_is_rejected_before_the_budget_is_spent(self, tmp_path):
        with pytest.raises(ConfigError, match="sample_num_rows"):
            make("DPflow_AdvGAN", sample_num_rows=-5)
        report = tmp_path / "privacy.json"
        syn = make("DPflow_AdvGAN", seed=0, epochs=1, privacy_report=str(report))
        syn.update_config({"sample_num_rows": -1})
        with pytest.raises(ConfigError, match="sample_num_rows"):
            syn.fit(make_data())
        assert syn.privacy == {}
        assert not report.exists()

    def test_budget_shares_must_leave_room_for_dp_sgd(self):
        with pytest.raises(ConfigError, match="share"):
            make("DPflow_AdvGAN", epsilon=1, share_histograms=0.5, share_copula=0.5)

    def test_keep_from_seed_must_name_declared_columns(self):
        with pytest.raises(ConfigError, match="keep_from_seed"):
            make("DPflow_AdvGAN", epsilon=1, keep_from_seed=["label"])


# 內建 Gaussian Copula(petsard_gaussian_copula.py)之常數欄位修正只在其自身程式路徑：
# DPflow_Copula 與 DPflow_AdvGAN 不呼叫其相關矩陣程式，固定種子下輸出逐值相同(含常數欄位之資料)
CONST_SPEC = SPEC + [
    {"name": "const_num", "type": "num", "integer": True},
    {"name": "const_cat", "type": "cat", "values": ["x", "y"]},
]


def make_constant_data(n: int = 400) -> pd.DataFrame:
    data = make_data(n, seed=1)
    data["const_num"] = 0
    data["const_cat"] = "x"
    return data


class TestBuiltinCopulaCodeIsNotUsed:
    @pytest.mark.parametrize("constant", [False, True], ids=["no_constant", "constant_columns"])
    @pytest.mark.parametrize(
        "method",
        [
            "DPflow_Copula",
            pytest.param("DPflow_AdvGAN", marks=pytest.mark.skipif(not HAS_ADVGAN, reason="needs the advgan extra")),
        ],
    )
    def test_same_output_without_the_builtin_copula_code(self, method, constant):
        from unittest.mock import patch

        from petsard.synthesizer.petsard_gaussian_copula import (
            PetsardGaussianCopulaSynthesizer,
        )

        data, spec = (make_constant_data(), CONST_SPEC) if constant else (make_data(), SPEC)
        extra = {"epochs": 1, "batch_size": 64} if method == "DPflow_AdvGAN" else {}
        _, first = run(method, data, column_spec=spec, seed=0, **extra)

        def not_used(*args, **kwargs):
            raise AssertionError("the built-in Gaussian Copula correlation code was used")

        with patch.object(PetsardGaussianCopulaSynthesizer, "_separate_constant_columns", not_used),                 patch.object(PetsardGaussianCopulaSynthesizer, "_regularize_correlation_matrix", not_used):
            _, second = run(method, data, column_spec=spec, seed=0, **extra)
        pd.testing.assert_frame_equal(first, second, check_exact=True)
