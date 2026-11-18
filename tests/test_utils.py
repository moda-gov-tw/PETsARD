"""Tests for the module path resolution of petsard.utils.

petsard.utils 模組路徑解析之測試。
"""

import logging

import pytest

from petsard.exceptions import UnableToLoadError
from petsard.utils import _resolve_module_path

LOGGER = logging.getLogger("PETsARD.test_utils")


class TestResolveModulePath:
    # 測試資料夾項目：與 module_path 組合後找到模組
    def test_directory_entry_is_joined_with_module_path(self, tmp_path):
        module = tmp_path / "my_module.py"
        module.write_text("x = 1\n", encoding="utf-8")
        resolved = _resolve_module_path("my_module.py", LOGGER, [str(tmp_path)])
        assert resolved == str(module)

    # 測試指向 .py 檔的項目：檔名與 module_path 相同時視為模組檔
    @pytest.mark.parametrize(
        "module_path",
        ["my_module.py", "some_dir/my_module.py", "some_dir\\my_module.py"],
    )
    def test_python_file_entry_with_the_same_name_is_a_candidate(
        self, tmp_path, module_path
    ):
        module = tmp_path / "my_module.py"
        module.write_text("x = 1\n", encoding="utf-8")
        resolved = _resolve_module_path(module_path, LOGGER, [str(module)])
        assert resolved == str(module)

    # 測試指向其他 .py 檔的項目：檔名不同時不當作模組檔，找不到時引發 UnableToLoadError
    def test_python_file_entry_with_another_name_is_skipped(self, tmp_path):
        module = tmp_path / "my_module.py"
        module.write_text("x = 1\n", encoding="utf-8")
        with pytest.raises(UnableToLoadError) as exc_info:
            _resolve_module_path("no_such_module.py", LOGGER, [str(module)])
        assert "not found" in exc_info.value.message
        assert "my_module.py" not in exc_info.value.message

    # 測試非 .py 的檔案(例如 pytest.exe 或 zip)與非字串項目：不當作模組檔，找不到時引發 UnableToLoadError
    def test_other_files_and_non_str_entries_are_skipped(self, tmp_path):
        launcher = tmp_path / "pytest.exe"
        launcher.write_bytes(b"MZ")
        archive = tmp_path / "python311.zip"
        archive.write_bytes(b"PK")
        with pytest.raises(UnableToLoadError) as exc_info:
            _resolve_module_path(
                "no_such_module.py", LOGGER, [str(launcher), str(archive), None, 3]
            )
        assert "not found" in exc_info.value.message
        assert "pytest.exe" not in exc_info.value.message
        assert "python311.zip" not in exc_info.value.message
