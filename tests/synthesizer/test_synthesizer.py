import subprocess
import sys
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from petsard.exceptions import ConfigError, MissingDependencyError, UncreatedError
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


# 測試 AdvGAN 選用依賴
class TestAdvGANOptionalDependency:
    # 測試未安裝 Opacus 時選用 AdvGAN 會引發錯誤
    def test_advgan_without_opacus_raises_missing_dependency(self):
        with patch.dict(sys.modules, {"opacus": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                Synthesizer(method="AdvGAN")
        assert "petsard[advgan]" in str(exc_info.value)

    # 測試未安裝 Opacus 時可匯入 PETsARD
    def test_import_petsard_without_opacus(self):
        code = "import sys; sys.modules['opacus'] = None; import petsard; print('ok')"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert result.stdout.strip().endswith("ok")

    # 測試延遲載入 AdvGAN 合成器
    def test_advgan_lazy_loads_synthesizer_class(self):
        pytest.importorskip("opacus")
        from petsard.synthesizer.DP_AdvGAN import AdvGANSynthesizer

        synthesizer = Synthesizer(method="AdvGAN")
        assert synthesizer.config.method_code == SynthesizerMap.ADVGAN
        assert (
            synthesizer._get_synthesizer_class(SynthesizerMap.ADVGAN)
            is AdvGANSynthesizer
        )
