"""
PETsARD Notebook Loader - Auto-import Version
PETsARD Notebook 載入器 - 自動導入版本

Copy the following code to the beginning of your notebook:
複製以下程式碼到你的 notebook 開頭：

=== Copy this to notebook / 複製這段到 notebook ===
import os  # noqa: I001
import sys
from pathlib import Path

# Handle utils.py for Colab
if "COLAB_GPU" in os.environ:
    import urllib.request

    demo_utils_url = (
        "https://raw.githubusercontent.com/moda-gov-tw/PETsARD/main/demo/demo_utils.py"
    )
    # Download and save to file instead of exec
    demo_utils_content = urllib.request.urlopen(demo_utils_url).read().decode("utf-8")
    with open("demo_utils.py", "w") as f:
        f.write(demo_utils_content)
    print("✅ demo_utils.py downloaded")

else:
    # demo_utils.py search for local
    for p in [Path.cwd()] + list(Path.cwd().parents)[:10]:
        utils_path = p / "demo_utils.py"
        if utils_path.exists() and "demo" in str(utils_path):
            sys.path.insert(0, str(p))
            break
=== End of copy / 複製結束 ===
"""

# ============================================================================
#                    Module Imports / 模組導入
# ============================================================================

import os
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

import requests

# Suppress NumPy warnings for invalid values in operations
# 抑制 NumPy 在運算中遇到無效值的警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# ============================================================================
#              PETsARD Setup Manager / PETsARD 設定管理器
# ============================================================================


class PETsARDSetup:
    """
    PETsARD Environment Setup Manager / PETsARD 環境設定管理器

    Handles environment detection, version management, package installation, file downloads and other basic configurations.
    負責處理環境檢測、版本管理、套件安裝、檔案下載等基礎設定。
    """

    # Class variable to remember original notebook path across runs / 類別變數記住原始 notebook 路徑
    _notebook_original_path = None

    def __init__(self, notebook_path=None):
        """
        Initialize environment setup / 初始化環境設定

        Args:
            notebook_path: Path to Jupyter notebook (optional) / Jupyter notebook 的路徑（可選）
        """
        self.is_colab = "COLAB_GPU" in os.environ
        self.demo_utils_file = "demo_utils.py"

        # Get the actual notebook location using multiple methods / 使用多種方法取得實際的 notebook 位置
        self.original_cwd = self._detect_notebook_location(notebook_path)

        # Store the original path for reference / 儲存原始路徑供參考
        PETsARDSetup._notebook_original_path = self.original_cwd

        self.demo_root = self.find_demo_root(self.original_cwd)
        self.original_subfolder = self._calculate_subfolder()

        # Switch working directory to demo if needed / 如果需要，切換工作目錄到 demo
        self._setup_working_directory()

    def _detect_notebook_location(self, notebook_path=None):
        """
        Detect the actual notebook location using multiple methods / 使用多種方法偵測實際的 notebook 位置

        Priority order / 優先順序:
        1. Explicitly provided path / 明確提供的路徑
        2. Get from IPython current notebook / 從 IPython 取得當前 notebook
        3. Use Jupyter API if available / 如果可用，使用 Jupyter API
        4. Check stored path from previous run / 檢查之前執行儲存的路徑
        5. Use current working directory / 使用當前工作目錄
        """
        current_path = Path.cwd().resolve()

        # Method 1: Use explicitly provided path / 使用明確提供的路徑
        if notebook_path:
            return Path(notebook_path).resolve()

        # Method 2: Try to get current notebook path from IPython / 嘗試從 IPython 取得當前 notebook 路徑
        try:
            from IPython import get_ipython

            ipython = get_ipython()

            if ipython and hasattr(ipython, "user_ns"):
                # Try to get notebook name from IPython magic / 嘗試從 IPython magic 取得 notebook 名稱
                import re

                # Get the notebook name from Jupyter session / 從 Jupyter session 取得 notebook 名稱
                try:
                    import ipykernel
                    import requests
                    from notebook.notebookapp import list_running_servers

                    # Get kernel id / 取得 kernel id
                    kernel_id = re.search(
                        "kernel-(.*).json", ipykernel.get_connection_file()
                    ).group(1)

                    # Find the notebook using kernel id / 使用 kernel id 尋找 notebook
                    servers = list(list_running_servers())
                    for server in servers:
                        try:
                            response = requests.get(
                                f"{server['url']}api/sessions",
                                headers={
                                    "Authorization": f"token {server.get('token', '')}"
                                },
                                timeout=30,
                            )
                            if response.status_code == 200:
                                sessions = response.json()
                                for sess in sessions:
                                    if sess["kernel"]["id"] == kernel_id:
                                        # Found the notebook! / 找到 notebook 了！
                                        nb_path = sess["notebook"]["path"]
                                        full_path = (
                                            Path(server["notebook_dir"]) / nb_path
                                        )
                                        if full_path.exists():
                                            return full_path.parent.resolve()
                        except Exception:
                            continue
                except Exception:
                    pass
        except Exception:
            pass

        # Method 3: Search for .ipynb files in parent directories / 在父目錄中搜尋 .ipynb 檔案
        # This works even if we're in demo directory / 即使在 demo 目錄中也能運作
        search_paths = []

        # If we're in demo, start searching from parent / 如果在 demo，從父目錄開始搜尋
        if current_path.name == "demo":
            search_paths.extend(
                [current_path.parent] + list(current_path.parent.parents)[:3]
            )
        else:
            search_paths.extend([current_path] + list(current_path.parents)[:5])

        # Method 4: Try alternative Jupyter API approach / 嘗試替代的 Jupyter API 方法
        try:
            from IPython import get_ipython

            ipython = get_ipython()

            if ipython and hasattr(ipython, "kernel"):
                # Try to use Jupyter API / 嘗試使用 Jupyter API
                try:
                    import requests
                    from notebook import notebookapp

                    # Get list of running notebook servers / 取得執行中的 notebook 伺服器清單
                    servers = list(notebookapp.list_running_servers())
                    for server in servers:
                        # Query each server for notebook list / 查詢每個伺服器的 notebook 清單
                        response = requests.get(
                            f"{server['url']}api/sessions",
                            headers={
                                "Authorization": f"token {server.get('token', '')}"
                            },
                            timeout=30,
                        )
                        if response.status_code == 200:
                            sessions = response.json()
                            for session in sessions:
                                # Check if this is our notebook / 檢查這是否是我們的 notebook
                                nb_path = session.get("notebook", {}).get("path", "")
                                if nb_path:
                                    full_path = Path(server["notebook_dir"]) / nb_path
                                    if full_path.exists():
                                        return full_path.parent.resolve()
                except Exception:
                    pass

        except Exception:
            pass

        # Method 5: Check stored path from previous run / 檢查之前執行儲存的路徑
        if (
            hasattr(PETsARDSetup, "_notebook_original_path")
            and PETsARDSetup._notebook_original_path
        ):
            stored_path = PETsARDSetup._notebook_original_path
            # Verify the path still makes sense / 驗證路徑是否仍然合理
            if stored_path.exists() and "demo" in str(stored_path):
                return stored_path

        # Method 6: Last resort - use current directory / 最後手段 - 使用當前目錄
        # If we're in demo root and can't detect, return current path / 如果在 demo 根目錄且無法偵測，回傳當前路徑
        return current_path

    def _calculate_subfolder(self):
        """Calculate notebook's subfolder path relative to demo / 計算 notebook 相對於 demo 的子資料夾路徑"""
        if self.demo_root and self.original_cwd:
            try:
                relative = self.original_cwd.relative_to(self.demo_root)
                return str(relative) if str(relative) != "." else None
            except ValueError:
                pass
        return None

    def _setup_working_directory(self):
        """Set working directory (switch to demo) / 設定工作目錄（切換到 demo）"""
        if self.demo_root and self.original_cwd != self.demo_root:
            try:
                os.chdir(self.demo_root)
                if not self.is_colab:  # Colab 不需要顯示這些訊息
                    print(
                        f"✅ Changed working directory to demo: {self._get_privacy_path(self.demo_root)}"
                    )
                    if self.original_subfolder:
                        print(
                            f"   📁 Notebook location: demo/{self.original_subfolder}/"
                        )
                        print("   🔍 YAML search priority: ")
                        print(f"      1. demo/{self.original_subfolder}/")
                        print("      2. demo/")
                    print("   💾 Output files will be saved in: demo/")
            except Exception as e:
                if not self.is_colab:
                    print(f"⚠️ Could not change to demo directory: {e}")

    # ========== Path Search Methods / 路徑搜尋相關方法 ==========

    def find_demo_root(self, current_path: Path | None = None) -> Path | None:
        """
        Find demo root directory / 找到 demo 根目錄

        Args:
            current_path: Starting search path, defaults to current working directory / 起始搜尋路徑，預設為當前工作目錄

        Returns:
            Path object of demo root directory, None if not found / demo 根目錄的 Path 物件，找不到則回傳 None
        """
        if current_path is None:
            current_path = Path.cwd()

        for parent in [current_path] + list(current_path.parents):
            if (parent / self.demo_utils_file).exists() and parent.name == "demo":
                return parent
        return None

    def find_project_root(self) -> Path | None:
        """
        Find project root directory (with pyproject.toml) / 找到專案根目錄 (含 pyproject.toml)

        Returns:
            Path object of project root directory, None if not found / 專案根目錄的 Path 物件，找不到則回傳 None
        """
        current_path = Path.cwd()
        for parent in [current_path] + list(current_path.parents):
            if (parent / "pyproject.toml").exists():
                return parent
        return None

    # ========== Version Management Methods / 版本管理相關方法 ==========

    def get_version(self) -> str:
        """
        Get PETsARD version / 取得 PETsARD 版本

        Returns:
            Version string, returns "unknown" if unable to get / 版本字串，無法取得時回傳 "unknown"
        """
        # 優先嘗試從已安裝的套件取得版本
        try:
            import importlib.metadata

            return importlib.metadata.version("petsard")
        except (ImportError, Exception):
            pass

        # 回退：從 pyproject.toml 讀取版本
        try:
            import tomllib

            project_root = self.find_project_root()
            if project_root:
                with open(project_root / "pyproject.toml", "rb") as f:
                    data = tomllib.load(f)
                return data.get("project", {}).get("version", "unknown")
        except (ImportError, FileNotFoundError, Exception):
            pass

        return "unknown"

    def check_python_version(self) -> None:
        """
        Check if Python version meets requirements / 檢查 Python 版本是否符合要求

        Raises:
            RuntimeError: When Python version doesn't meet requirements / Python 版本不符合要求時
        """
        version_str = ".".join(map(str, sys.version_info[:2]))

        if sys.version_info < (3, 10):  # noqa: UP036
            raise RuntimeError(f"Requires Python 3.10 or 3.11. Current: {version_str}")
        elif sys.version_info >= (3, 12):
            raise RuntimeError(
                f"Python 3.12+ not yet supported. Current: {version_str}"
            )

    # ========== Installation and Download Methods / 安裝與下載相關方法 ==========

    def install_petsard(self, branch: str = "main") -> None:
        """
        Install PETsARD package / 安裝 PETsARD 套件

        Args:
            branch: GitHub branch name, defaults to "main" / GitHub 分支名稱，預設為 "main"
        """
        # 設定環境變數
        os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        # Colab 環境中 pip 已預先安裝，不需要 ensurepip
        if not self.is_colab:
            # 本地環境：確保 pip 已安裝
            try:
                subprocess.run(
                    [sys.executable, "-m", "ensurepip"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError:
                # 如果 ensurepip 失敗，假設 pip 已經安裝
                pass

        if self.is_colab:
            # Colab: 從 GitHub 安裝
            print(
                f"📦 Installing PETsARD from branch '{branch}' with [all] dependencies..."
            )
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f"git+https://github.com/moda-gov-tw/PETsARD.git@{branch}#egg=petsard[all]",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("✅ PETsARD installed successfully")
            except subprocess.CalledProcessError as e:
                print("❌ Installation failed!")
                print(f"Error: {e}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                raise
        else:
            # 本地: 可編輯安裝
            project_root = self.find_project_root()
            if not project_root:
                raise FileNotFoundError(
                    "Could not find project root with pyproject.toml"
                )

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    f"{project_root}[all]",
                    "-q",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

    def download_file(self, repo_path: str, branch: str = "main") -> bool:
        """
        下載檔案 (僅 Colab 環境)

        Args:
            repo_path: 在 GitHub repo 中的檔案路徑
            branch: GitHub 分支名稱

        Returns:
            下載成功回傳 True，失敗回傳 False
        """
        if not self.is_colab:
            # 本地環境：檢查檔案是否存在
            return Path(Path(repo_path).name).exists()

        # Colab 環境：下載檔案
        local_file = Path(repo_path).name
        file_url = (
            f"https://raw.githubusercontent.com/moda-gov-tw/PETsARD/{branch}/{repo_path}"
        )

        try:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            with open(local_file, "w") as f:
                f.write(response.text)
            return True
        except Exception:
            return False

    # ========== YAML File Processing Methods / YAML 檔案處理相關方法 ==========

    def get_yaml_path(self, yaml_file: str, branch: str = "main") -> Path:
        """
        Get YAML file path with auto-detection and smart search / 取得 YAML 檔案路徑，自動偵測子目錄並智慧搜尋

        Args:
            yaml_file: YAML filename / YAML 檔案名稱
            branch: GitHub branch name (for Colab) / GitHub 分支名稱 (Colab 環境用)

        Returns:
            Complete path to YAML file / YAML 檔案的完整路徑

        Raises:
            FileNotFoundError: When YAML file not found / 找不到 YAML 檔案時
        """
        # 自動偵測 subfolder
        subfolder = self._auto_detect_subfolder()

        if self.is_colab:
            return self._get_yaml_path_colab(yaml_file, branch, subfolder)
        else:
            return self._get_yaml_path_local(yaml_file, subfolder)

    def _auto_detect_subfolder(self) -> str | None:
        """
        Get subfolder path relative to demo / 取得子資料夾路徑（相對於 demo/）
        """
        return self.original_subfolder

    def _get_yaml_path_colab(
        self, yaml_file: str, branch: str, subfolder: str | None
    ) -> Path:
        """Colab 環境的 YAML 檔案處理"""
        import tempfile

        yaml_url = (
            f"https://raw.githubusercontent.com/moda-gov-tw/PETsARD/{branch}/demo/"
            f"{subfolder + '/' if subfolder else ''}{yaml_file}"
        )
        response = requests.get(yaml_url, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            tmp.write(response.text)
            return Path(tmp.name)

    def _get_yaml_path_local(self, yaml_file: str, subfolder: str | None) -> Path:
        """
        Local YAML file processing with smart search / 本地環境的 YAML 檔案處理，支援智慧搜尋
        Always use absolute paths to avoid working directory issues / 總是使用絕對路徑以避免工作目錄問題
        """
        if not self.demo_root:
            raise FileNotFoundError("Could not find demo directory")

        # Build search paths list - all using absolute paths / 建立搜尋路徑清單 - 全部使用絕對路徑
        search_paths = []

        # 1. Highest priority: search in original notebook location / 最優先：在原始 notebook 位置搜尋
        if self.original_cwd:
            original_yaml = (self.original_cwd / yaml_file).resolve()
            if original_yaml not in search_paths:
                search_paths.append(original_yaml)

        # 2. If subfolder detected, search in that subfolder / 如果偵測到子資料夾，在該子資料夾搜尋
        if subfolder and self.demo_root:
            # Build paths from deep to shallow / 從深到淺建立路徑
            subfolder_path = Path(subfolder)

            # a. Full subfolder path / 完整子資料夾路徑
            full_subfolder_yaml = (self.demo_root / subfolder / yaml_file).resolve()
            if full_subfolder_yaml not in search_paths:
                search_paths.append(full_subfolder_yaml)

            # b. Parent paths of subfolder / 子資料夾的父路徑
            current = subfolder_path
            while current.parts:
                current = current.parent
                if str(current) != ".":
                    parent_yaml = (self.demo_root / current / yaml_file).resolve()
                    if parent_yaml not in search_paths:
                        search_paths.append(parent_yaml)
                else:
                    break

        # 3. Search in current working directory / 在當前工作目錄搜尋
        current_yaml = (Path.cwd() / yaml_file).resolve()
        if current_yaml not in search_paths:
            search_paths.append(current_yaml)

        # 4. Finally search in demo root / 最後在 demo 根目錄搜尋
        if self.demo_root:
            demo_yaml = (self.demo_root / yaml_file).resolve()
            if demo_yaml not in search_paths:
                search_paths.append(demo_yaml)

        # 移除重複路徑，保持順序（使用 resolve() 來比較絕對路徑）
        unique_paths = []
        seen = set()
        for path in search_paths:
            try:
                path_resolved = path.resolve()
                path_str = str(path_resolved)
                if path_str not in seen and path_resolved.parent.exists():
                    unique_paths.append(path)
                    seen.add(path_str)
            except (OSError, RuntimeError):
                # 處理無效路徑
                continue

        # 搜尋 YAML 檔案
        for yaml_path in unique_paths:
            if yaml_path.exists():
                # 回傳絕對路徑，確保在切換目錄後仍能正確存取
                return yaml_path.resolve()

        # 找不到檔案，提供詳細錯誤訊息
        self._raise_yaml_not_found_error(yaml_file, unique_paths)

    def _raise_yaml_not_found_error(
        self, yaml_file: str, search_paths: list[Path]
    ) -> None:
        """產生找不到 YAML 檔案的錯誤訊息"""
        # 只顯示前 3 個搜尋路徑
        tried_paths = "\n  - ".join(self._get_privacy_path(p) for p in search_paths[:3])
        raise FileNotFoundError(
            f"YAML file '{yaml_file}' not found. Searched in:\n  - {tried_paths}"
        )

    @staticmethod
    def _get_privacy_path(path: Path) -> str:
        """取得隱私友善的路徑字串"""
        path_str = Path(path).as_posix()
        if "petsard/" in path_str:
            return "petsard/" + path_str.split("petsard/", 1)[1]
        return path.name


# ============================================================================
#                    Helper Functions / 輔助函數
# ============================================================================


def display_yaml_info(
    yaml_path: Path | list[Path] | None,
    show_content: bool = True,
) -> None:
    """
    Display YAML file information (public function) / 顯示 YAML 檔案資訊（公開函數）

    Args:
        yaml_path: YAML file path(s) - can be single Path, list of Paths, or None / YAML 檔案路徑 - 可以是單個 Path、Path 清單或 None
        show_content: Whether to show file content / 是否顯示檔案內容
    """
    if yaml_path is None:
        print("ℹ️ No YAML file path provided / 未提供 YAML 檔案路徑")
        return

    # Handle single path / 處理單個路徑
    if isinstance(yaml_path, Path):
        yaml_paths = [yaml_path]
    elif isinstance(yaml_path, list):
        yaml_paths = yaml_path
    else:
        yaml_paths = [Path(yaml_path)]

    print("=" * 60)
    print("📋 YAML Configuration Files / YAML 設定檔案")
    print("=" * 60)

    for i, path in enumerate(yaml_paths, 1):
        # Get privacy-friendly path / 取得隱私友善的路徑
        privacy_path = PETsARDSetup._get_privacy_path(path)

        if len(yaml_paths) > 1:
            print(f"\n📄 File {i}/{len(yaml_paths)}: {path.name}")
        else:
            print(f"\n📄 File: {path.name}")

        print(f"📁 Path: {privacy_path}")

        if show_content:
            try:
                with open(path) as f:
                    content = f.read()
                    print("\n⚙️ Configuration content / 設定內容:")
                    print("-" * 40)
                    print(content)
                    if i < len(yaml_paths):
                        print("-" * 40)
            except Exception as e:
                print(f"❌ Failed to read content: {e}")

    print("=" * 60)


def display_results(results: dict, max_rows: int = 3) -> None:
    """
    Display execution results in a formatted way / 以格式化方式顯示執行結果

    Handles DataFrame results by showing only first few rows / 處理 DataFrame 結果，只顯示前幾行

    Args:
        results: Dictionary from exec_now.get_result() / 來自 exec_now.get_result() 的字典
        max_rows: Maximum rows to display for DataFrames / DataFrame 顯示的最大行數

    Example:
        >>> results = exec_now.get_result()
        >>> display_results(results)
    """
    if not results:
        print("ℹ️ No results to display / 沒有結果可顯示")
        return

    print("=" * 80)
    print("📊 Execution Results / 執行結果")
    print("=" * 80)

    for i, (key, value) in enumerate(results.items(), 1):
        print(f"\n[{i}] {key}")
        print("-" * 60)

        # Special handling for validation results / 特殊處理驗證結果
        if isinstance(value, dict) and "Reporter" in key:
            # This is likely a Reporter result
            print("📋 Reporter output / Reporter 輸出")

            # Check if validation report was saved
            if not value or all(
                v is None or (isinstance(v, dict) and not v) for v in value.values()
            ):
                print(
                    "✅ Validation report saved to CSV file / 驗證報告已保存為 CSV 檔案"
                )
                print("📁 Check the demo directory for petsard[Validation]_*.csv files")
                print("   查看 demo 目錄中的 petsard[Validation]_*.csv 檔案")
                continue

            # Display non-empty Reporter results
            for report_key, report_value in value.items():
                # Check if value is not None and not empty
                # For DataFrames, use .empty property; for other types, use truthiness
                is_not_empty = False
                if report_value is not None:
                    if hasattr(report_value, "empty"):
                        # It's a DataFrame or Series
                        is_not_empty = not report_value.empty
                    elif isinstance(report_value, dict):
                        is_not_empty = bool(report_value)
                    else:
                        is_not_empty = True

                if is_not_empty:
                    print(f"  • {report_key}: {type(report_value).__name__}")
            continue

        # Check if value is a pandas DataFrame / 檢查是否為 pandas DataFrame
        if hasattr(value, "head") and hasattr(value, "shape"):
            # It's a DataFrame / 這是 DataFrame
            rows, cols = value.shape
            print(f"📈 DataFrame: {rows:,} rows × {cols} columns")
            print(
                f"📋 Showing first {min(max_rows, rows)} rows / 顯示前 {min(max_rows, rows)} 行:\n"
            )

            # Display first few rows / 顯示前幾行
            display_df = value.head(max_rows)
            print(display_df.to_string())

            if rows > max_rows:
                print(
                    f"\n... ({rows - max_rows:,} more rows) / ... (還有 {rows - max_rows:,} 行)"
                )

            # Show column info / 顯示欄位資訊
            print(f"\n📝 Columns / 欄位: {', '.join(value.columns.tolist())}")

        elif isinstance(value, dict):
            # It's a dictionary / 這是字典
            print(
                f"📦 Dictionary with {len(value)} keys / 包含 {len(value)} 個鍵的字典"
            )

            # Separate DataFrame and non-DataFrame items / 分離 DataFrame 和非 DataFrame 項目
            df_items = []
            non_df_items = []

            for k, v in value.items():
                if hasattr(v, "head") and hasattr(v, "shape"):
                    df_items.append((k, v))
                else:
                    non_df_items.append((k, v))

            # Display DataFrames first with full details / 先顯示 DataFrames 的完整資訊
            for k, v in df_items:
                rows, cols = v.shape
                print(f"\n  • {k}: DataFrame ({rows:,} rows × {cols} columns)")
                print(
                    f"    📋 Showing first {min(max_rows, rows)} rows / 顯示前 {min(max_rows, rows)} 行:"
                )

                # Display first few rows with indentation / 顯示前幾行（帶縮排）
                display_df = v.head(max_rows)
                df_string = display_df.to_string()
                # Add indentation to each line / 為每行添加縮排
                indented_df = "\n".join(f"    {line}" for line in df_string.split("\n"))
                print(indented_df)

                if rows > max_rows:
                    print(
                        f"    ... ({rows - max_rows:,} more rows) / ... (還有 {rows - max_rows:,} 行)"
                    )

                # Show column info / 顯示欄位資訊
                print(f"    📝 Columns / 欄位: {', '.join(v.columns.tolist())}")

            # Display non-DataFrame items (only first 5) / 顯示非 DataFrame 項目（只顯示前 5 個）
            for k, v in non_df_items[:5]:
                print(f"  • {k}: {type(v).__name__}")

            # Show count of remaining non-DataFrame items / 顯示剩餘非 DataFrame 項目數
            if len(non_df_items) > 5:
                print(
                    f"  ... and {len(non_df_items) - 5} more non-DataFrame keys / ... 還有 {len(non_df_items) - 5} 個非 DataFrame 鍵"
                )

        elif isinstance(value, (list, tuple)):
            # It's a list or tuple / 這是列表或元組
            type_name = "List" if isinstance(value, list) else "Tuple"
            print(
                f"📝 {type_name} with {len(value)} items / 包含 {len(value)} 個項目的{type_name}"
            )
            for item in value[:3]:
                print(f"  • {repr(item)[:100]}")
            if len(value) > 3:
                print(
                    f"  ... and {len(value) - 3} more items / ... 還有 {len(value) - 3} 個項目"
                )

        elif isinstance(value, str):
            # It's a string / 這是字串
            if len(value) > 200:
                print(
                    f"📄 String ({len(value)} characters) / 字串（{len(value)} 個字元）:"
                )
                print(value[:200] + "...")
            else:
                print(f"📄 String: {value}")

        else:
            # Other types / 其他類型
            print(f"🔢 {type(value).__name__}: {repr(value)[:200]}")

    print("\n" + "=" * 80)
    print(f"✅ Total results / 總結果數: {len(results)}")
    print("=" * 80)


def _display_yaml_info(
    yaml_path: Path,
    yaml_name: str,
    index: int | None = None,
    total: int | None = None,
) -> None:
    """
    Internal function for displaying YAML info (used by quick_setup) / 內部函數用於顯示 YAML 資訊（供 quick_setup 使用）

    This is kept for backward compatibility but not called in quick_setup by default
    這是為了向後相容而保留，但預設不在 quick_setup 中呼叫
    """
    # Get privacy-friendly path / 取得隱私友善的路徑
    privacy_path = PETsARDSetup._get_privacy_path(yaml_path)

    # Display path / 顯示路徑
    if index and total:
        print(f"📄 YAML path ({index}/{total}): {privacy_path}")
    else:
        print(f"📄 YAML path: {privacy_path}")

    # Display content / 顯示內容
    try:
        with open(yaml_path) as f:
            content = f.read()
            if index and total:
                print(f"⚙️ Configuration content ({index}/{total}) - {yaml_name}:")
            else:
                print("⚙️ Configuration content:")
            print(content)
            if index and total and index < total:
                print("---")
    except Exception as e:
        print(f"❌ Failed to read YAML content: {e}")


def _load_benchmark_data(benchmark_data: list[str]) -> None:
    """
    載入基準資料集

    Args:
        benchmark_data: 要載入的基準資料集名稱清單
    """
    from petsard.adapter import LoaderAdapter
    from petsard.exceptions import BenchmarkDatasetsError, UnsupportedMethodError
    from petsard.loader.benchmarker import BenchmarkerConfig, BenchmarkerRequests

    for benchmark in benchmark_data:
        try:
            # 檢查是否為 schema 檔案（通常包含 _schema 後綴）
            if "_schema" in benchmark:
                # Schema 檔案只下載，不載入（因為 Loader 不支援 YAML 格式）
                try:
                    config = BenchmarkerConfig(
                        benchmark_name=benchmark,
                        filepath_raw=f"benchmark://{benchmark}",
                    )
                    benchmarker = BenchmarkerRequests(config.get_benchmarker_config())
                    benchmarker.download()
                    print(f"✅ Downloaded benchmark schema: {benchmark}")
                except (BenchmarkDatasetsError, UnsupportedMethodError) as e:
                    print(f"⚠️ Failed to download benchmark schema {benchmark}: {e}")
            else:
                # 資料檔案正常載入
                config = {"filepath": f"benchmark://{benchmark}"}
                loader_adapter = LoaderAdapter(config)
                loader_adapter._run({})  # 執行載入
                print(f"✅ Loaded benchmark dataset: {benchmark}")
        except Exception as e:
            print(f"⚠️ Failed to load benchmark dataset {benchmark}: {e}")


def _download_example_files(
    setup: PETsARDSetup, example_files: list[str], branch: str
) -> None:
    """
    下載範例檔案

    Args:
        setup: PETsARDSetup 實例
        example_files: 要下載的檔案路徑清單
        branch: GitHub 分支名稱
    """
    action = "Downloading" if setup.is_colab else "Checking"
    print(f"📥 {action} example files:")

    for repo_path in example_files:
        local_file = Path(repo_path).name
        if setup.download_file(repo_path, branch):
            print(f"✅ {local_file}")
        else:
            status = "❌ Failed:" if setup.is_colab else "ℹ️ Expected:"
            print(f"{status} {local_file}")


def _display_environment_info(setup: PETsARDSetup, version: str) -> None:
    """
    顯示環境資訊

    Args:
        setup: PETsARDSetup 實例
        version: PETsARD 版本
    """
    # 取得時區資訊
    now = datetime.now()
    utc_offset = now.astimezone().utcoffset()
    offset_hours = int(utc_offset.total_seconds() / 3600)
    offset_str = f"UTC{offset_hours:+d}" if offset_hours != 0 else "UTC"

    # 顯示資訊
    print(f"🚀 PETsARD v{version}")
    print(f"📅 {now.strftime('%Y-%m-%d %H:%M:%S')} {offset_str}")


# ============================================================================
#                    Main Functions / 主要函數
# ============================================================================


def quick_setup(
    config_file: str | list[str] | None = None,
    benchmark_data: list[str] | None = None,
    petsard_branch: str = "main",
    example_files: list[str] | None = None,
    notebook_path: str | None = None,
    show_yaml_content: bool = False,  # New parameter / 新參數
) -> tuple[bool, str, str | list[str] | None, type]:
    """
    Quick setup for PETsARD environment / 快速設定 PETsARD 環境

    Provides one-stop environment setup including version check, package installation, file downloads, etc.
    提供一站式的環境設定，包括版本檢查、套件安裝、檔案下載等。

    Args:
        config_file: Configuration filename(s) - YAML or Python (single or multiple) / 設定檔名稱 - YAML 或 Python (單個或多個)
        benchmark_data: List of benchmark datasets to load / 要載入的基準資料集清單
        petsard_branch: PETsARD GitHub branch name, defaults to "main" / PETsARD GitHub 分支名稱，預設為 "main"
        example_files: List of example files to download / 要下載的範例檔案清單
        notebook_path: Path to Jupyter notebook (optional, auto-detected) / Jupyter notebook 的路徑（可選，自動偵測）
        show_yaml_content: Whether to display YAML content (default: False) / 是否顯示 YAML 內容（預設：False）

    Returns:
        Tuple containing / Tuple 包含:
        - is_colab: Whether in Colab environment / 是否為 Colab 環境
        - petsard_branch: PETsARD branch name used / 使用的 PETsARD 分支名稱
        - yaml_path: YAML file path(s) as string (single or multiple) / YAML 檔案路徑字串 (單個或多個)
        - Executor: PETsARD Executor class / PETsARD Executor 類別

    Examples:
        >>> # Single configuration file / 單個設定檔
        >>> is_colab, branch, config_path = quick_setup("config.yaml")
        >>>
        >>> # Display config info separately / 分別顯示設定資訊
        >>> display_yaml_info(config_path)

        >>> # Multiple configuration files with specific branch / 多個設定檔，指定特定分支
        >>> is_colab, branch, paths = quick_setup(
        ...     config_file=["config1.yaml", "config2.py"],
        ...     benchmark_data=["adult", "census"],
        ...     petsard_branch="develop"
        ... )
    """
    # Create PETsARDSetup instance with automatic path handling / 創建 PETsARDSetup 實例，自動處理路徑
    setup = PETsARDSetup(notebook_path=notebook_path)

    # 1. Environment check and package installation / 環境檢查與套件安裝
    setup.check_python_version()
    setup.install_petsard(petsard_branch)

    # 2. Display environment info / 顯示環境資訊
    version = setup.get_version()
    _display_environment_info(setup, version)

    # 3. Download example files (if needed) / 下載範例檔案（如果需要）
    if example_files:
        _download_example_files(setup, example_files, petsard_branch)

    # 4. Load benchmark datasets (if needed) / 載入基準資料集（如果需要）
    if benchmark_data:
        _load_benchmark_data(benchmark_data)

    # 5. Add notebook directory to Python path for module imports / 將 notebook 目錄加入 Python 路徑以供模組導入
    if setup.original_cwd and setup.original_cwd not in [Path(p) for p in sys.path]:
        # Add the notebook's original directory to sys.path / 將 notebook 的原始目錄加入 sys.path
        sys.path.insert(0, str(setup.original_cwd))
        if not setup.is_colab:
            print(
                f"🔧 Added to Python path: {setup._get_privacy_path(setup.original_cwd)}"
            )

    # 6. Process configuration files / 處理設定檔案
    yaml_path = None
    if config_file:
        # Auto-detect and display current subfolder location / 自動偵測並顯示當前子資料夾位置
        subfolder = setup._auto_detect_subfolder()
        if subfolder:
            print(f"📁 Processing configuration files from subfolder: {subfolder}")

        if isinstance(config_file, str):
            # Single configuration file (auto-detect path) / 單個設定檔（自動偵測路徑）
            yaml_path = setup.get_yaml_path(config_file, petsard_branch)
            print(f"✅ Found configuration: {setup._get_privacy_path(yaml_path)}")

            # Only show content if requested / 只在要求時顯示內容
            if show_yaml_content:
                _display_yaml_info(yaml_path, config_file)

        elif isinstance(config_file, list):
            # Multiple configuration files (auto-detect paths) / 多個設定檔（自動偵測路徑）
            yaml_path = []
            for i, single_config in enumerate(config_file, 1):
                single_path = setup.get_yaml_path(single_config, petsard_branch)
                yaml_path.append(single_path)
                print(
                    f"✅ Found configuration ({i}/{len(config_file)}): {setup._get_privacy_path(single_path)}"
                )

                # Only show content if requested / 只在要求時顯示內容
                if show_yaml_content:
                    _display_yaml_info(single_path, single_config, i, len(config_file))

    # Convert Path to string for Executor compatibility / 轉換 Path 為字串以相容 Executor
    if yaml_path:
        if isinstance(yaml_path, Path):
            yaml_path = str(yaml_path)
        elif isinstance(yaml_path, list):
            yaml_path = [str(p) for p in yaml_path]

    # Import Executor after petsard is installed / 在 petsard 安裝後導入 Executor
    from petsard import Executor

    return setup.is_colab, petsard_branch, yaml_path, Executor


# ============================================================================
#       Simplified Notebook Loading Prompt / 簡化的 Notebook 載入提示
# ============================================================================

# 當通過 exec() 執行時（例如在 Jupyter notebook 中）
if "__file__" not in globals():
    import os
    from pathlib import Path

    # 只顯示簡單的提示，不做任何全域變數操作
    _current_path = Path.cwd()
    _privacy_path = (
        _current_path.as_posix().split("petsard/")[-1]
        if "petsard/" in _current_path.as_posix()
        else _current_path.name
    )
    print(f"📂 Current working directory: {_privacy_path}")
    print("✅ PETsARD demo_utils loaded. Use quick_setup() to initialize.")
