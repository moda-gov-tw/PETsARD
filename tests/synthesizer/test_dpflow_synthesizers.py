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

    def test_budget_shares_must_leave_room_for_dp_sgd(self):
        with pytest.raises(ConfigError, match="share"):
            make("DPflow_AdvGAN", epsilon=1, share_histograms=0.5, share_copula=0.5)

    def test_keep_from_seed_must_name_declared_columns(self):
        with pytest.raises(ConfigError, match="keep_from_seed"):
            make("DPflow_AdvGAN", epsilon=1, keep_from_seed=["label"])
