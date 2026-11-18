import subprocess
import sys
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from petsard.exceptions import (
    ConfigError,
    MissingDependencyError,
    UnableToLoadError,
    UncreatedError,
    UnsupportedMethodError,
)
from petsard.synthesizer.synthesizer import Synthesizer, SynthesizerMap


# 測試 Synthesizer 基本功能
class TestSynthesizer:
    # 測試初始化功能 - 使用內建方法
    def test_initialization(self):
        synthesizer = Synthesizer(method="petsard-gaussian_copula")
        assert synthesizer.config.method == "petsard-gaussian_copula"
        assert synthesizer._impl is None

        synthesizer = Synthesizer(
            method="petsard-gaussian_copula", sample_num_rows=500
        )
        assert synthesizer.config.sample_num_rows == 500

    # 測試 SDV 方法需要依賴檢查
    def test_sdv_method_requires_dependency(self):
        with patch.dict(sys.modules, {"sdv": None}):
            with pytest.raises(MissingDependencyError):
                Synthesizer(method="sdv-single_table-gaussiancopula")

    # 測試 create 方法 - 使用內建方法
    def test_create_basic(self):
        synthesizer = Synthesizer(method="petsard-gaussian_copula")

        # 在 create 前，_impl 為 None
        assert synthesizer._impl is None

        # 模擬 _determine_sample_configuration 方法，避免依賴實際邏輯
        with patch.object(
            synthesizer,
            "_determine_sample_configuration",
            return_value=("Source data", 100),
        ):
            synthesizer.create()

            # 在 create 後，_impl 已設置
            assert synthesizer._impl is not None

    # 測試在未 create 前呼叫 fit 會引發 UncreatedError
    def test_fit_without_create(self):
        synthesizer = Synthesizer(method="petsard-gaussian_copula")
        with pytest.raises(UncreatedError):
            synthesizer.fit(data=pd.DataFrame())

    # 測試非 CUSTOM_DATA 方法但無資料時引發 ConfigError
    def test_fit_without_data_raises_error(self):
        synthesizer = Synthesizer(method="petsard-gaussian_copula")
        # 手動設置 _impl，避免使用 create
        synthesizer._impl = Mock()
        synthesizer.config.method_code = SynthesizerMap.PETSARD

        with pytest.raises(ConfigError):
            synthesizer.fit(data=None)

    # 測試 sample 在未 create 時返回空 DataFrame
    def test_sample_without_create(self):
        synthesizer = Synthesizer(method="petsard-gaussian_copula")
        result = synthesizer.sample()
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# 測試 DPflow_AdvGAN 選用依賴
class TestDPflowAdvGANOptionalDependency:
    # 測試未安裝 Opacus 時選用 DPflow_AdvGAN 會引發錯誤
    def test_dpflow_advgan_without_opacus_raises_missing_dependency(self):
        with patch.dict(sys.modules, {"opacus": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                Synthesizer(method="DPflow_AdvGAN")
        assert "petsard[advgan]" in str(exc_info.value)

    # 測試未安裝 Opacus 時可匯入 PETsARD
    def test_import_petsard_without_opacus(self):
        code = "import sys; sys.modules['opacus'] = None; import petsard; print('ok')"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip().endswith("ok")

    # 測試延遲載入 DPflow_AdvGAN 合成器
    def test_dpflow_advgan_lazy_loads_synthesizer_class(self):
        pytest.importorskip("opacus")
        pytest.importorskip("dp_accounting")
        from petsard.synthesizer.DPflow_AdvGAN import DPflowAdvGANSynthesizer

        synthesizer = Synthesizer(method="DPflow_AdvGAN")
        assert synthesizer.config.method_code == SynthesizerMap.DPFLOW_ADVGAN
        assert (
            synthesizer._get_synthesizer_class(SynthesizerMap.DPFLOW_ADVGAN)
            is DPflowAdvGANSynthesizer
        )

    # 測試方法名稱 AdvGAN 已不受支援，引發 UnsupportedMethodError 並列出可用方法
    def test_removed_advgan_method_is_unsupported(self):
        with pytest.raises(UnsupportedMethodError, match="Unsupported synthesizer method") as exc_info:
            Synthesizer(method="AdvGAN")
        assert "DPflow_AdvGAN" in exc_info.value.context["valid_methods"]
        assert "DPflow_AdvGAN" in exc_info.value.get_suggestion()
        # 訊息說明初版 AdvGAN 已移除並指向 DPflow_AdvGAN
        assert "has been removed" in exc_info.value.message
        assert "method: DPflow_AdvGAN" in exc_info.value.message

    # 測試其他不支援的方法名稱不附初版方法移除說明
    def test_unknown_method_has_no_removal_hint(self):
        with pytest.raises(UnsupportedMethodError) as exc_info:
            Synthesizer(method="no_such_method")
        assert "has been removed" not in exc_info.value.message


# 測試以 custom_method 指定已移除之初版方法模組時，錯誤訊息指出替代方法
class TestRemovedInitialModules:
    @staticmethod
    def _load(module_path: str, class_name: str):
        from petsard.synthesizer.custom_synthesizer import CustomSynthesizer

        return CustomSynthesizer(
            {
                "syn_method": "custom_method",
                "module_path": module_path,
                "class_name": class_name,
            }
        )

    # 測試初版 LLM 合成器模組：指向 DP_LLM_synthesizer
    def test_initial_llm_synthesizer_module_points_to_dp_llm(self):
        with pytest.raises(UnableToLoadError) as exc_info:
            self._load("petsard/synthesizer/LLM_synthesizer.py", "LLM_synthesizer")
        assert "not found" in exc_info.value.message
        assert "class_name: DP_LLM_synthesizer" in exc_info.value.message

    # 測試初版 AdvGAN 模組(含 Windows 路徑寫法)：指向 DPflow_AdvGAN
    @pytest.mark.parametrize(
        "module_path",
        ["petsard/synthesizer/DP_AdvGAN.py", r"petsard\synthesizer\AdvGAN.py"],
    )
    def test_initial_advgan_module_points_to_dpflow_advgan(self, module_path):
        with pytest.raises(UnableToLoadError) as exc_info:
            self._load(module_path, "AdvGANSynthesizer")
        assert "method: DPflow_AdvGAN" in exc_info.value.message

    # 測試其他找不到的模組維持原有訊息
    def test_other_missing_module_keeps_message(self):
        with pytest.raises(UnableToLoadError) as exc_info:
            self._load("no_such_dir/my_synthesizer.py", "MySynthesizer")
        assert "has been removed" not in exc_info.value.message
