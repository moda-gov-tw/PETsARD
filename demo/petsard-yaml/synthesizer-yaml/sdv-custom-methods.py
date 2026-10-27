"""
SDV Custom Methods Demo - SDV 自訂方法示範
Demonstrates how to wrap SDV methods as PETsARD custom_method
示範如何將 SDV 方法封裝為 PETsARD custom_method

This script provides wrapper classes for SDV synthesizers that can be used
as custom methods in PETsARD, allowing users to leverage SDV's powerful
synthetic data generation capabilities without depending on SDV as a core dependency.
本腳本提供 SDV 合成器的封裝類別，可作為 PETsARD 的自訂方法使用，
讓使用者能夠利用 SDV 強大的合成資料生成能力，而無需將 SDV 作為核心依賴。

Requirements 需求:
    pip install sdv

Supported Methods 支援的方法:
    - SDV_GaussianCopula: Fast statistical method 快速、基於統計的方法
    - SDV_CTGAN: High-quality GAN method 高品質 GAN 方法
    - SDV_CopulaGAN: Hybrid statistical + GAN method 混合統計和 GAN 的方法
    - SDV_TVAE: VAE-based method 基於 VAE 的方法
"""

import importlib.util
import logging
import time
import warnings
from typing import Any

import pandas as pd

from petsard.metadater import Schema


def check_sdv_installed():
    """
    Check if SDV is installed 檢查 SDV 是否已安裝

    Raises:
        ImportError: If SDV is not installed 如果 SDV 未安裝
    """
    if importlib.util.find_spec("sdv") is None:
        error_msg = """
        SDV package is not installed! SDV 套件未安裝！

        Please install SDV using: 請使用以下指令安裝 SDV：
            pip install sdv

        Or install a specific version: 或安裝特定版本：
            pip install sdv==1.15.0

        For more information: 更多資訊請參考：https://sdv.dev/
        """
        raise ImportError(error_msg)
    return True


def schema_to_sdv(schema: Schema, data: pd.DataFrame = None) -> dict[str, Any]:
    """
    Convert PETsARD Schema to SDV metadata format 將 PETsARD Schema 轉換為 SDV metadata 格式

    Args:
        schema: PETsARD Schema object PETsARD Schema 物件
        data: Optional training data to extract category values 可選的訓練數據以提取類別值

    Returns:
        dict: SDV metadata dictionary SDV metadata 格式的字典
    """
    # 🔍 DIAGNOSTIC: Track conversion process
    print("\n🔍 DIAGNOSTIC: schema_to_sdv() called")
    print(f"  Schema has {len(schema.attributes)} attributes")
    if data is not None:
        print(f"  Data provided: shape {data.shape}")

    sdv_metadata = {"columns": {}, "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1"}

    for attr_name, attribute in schema.attributes.items():
        # 🔍 DIAGNOSTIC: 顯示每個欄位的 schema type
        print(
            f"  {attr_name}: schema.type={getattr(attribute, 'type', 'N/A')}, category={getattr(attribute, 'category', 'N/A')}"
        )

        # Get actual dtype from data if available
        actual_dtype = None
        if data is not None and attr_name in data.columns:
            actual_dtype = data[attr_name].dtype
            print(f"    → actual dtype in data: {actual_dtype}")

        sdtype = _map_attribute_to_sdv_type(attribute, actual_dtype=actual_dtype)
        print(f"    → SDV sdtype: {sdtype}")
        col_info = {"sdtype": sdtype}

        # If data is provided and column is categorical, extract values
        # 如果提供了數據且欄位是類別型，提取類別值
        if data is not None and attr_name in data.columns and sdtype == "categorical":
            unique_vals = data[attr_name].dropna().unique()
            # Only include values list for reasonable cardinality (< 100)
            # 只對合理基數（< 100）的欄位包含值列表
            if len(unique_vals) < 100:
                col_info["values"] = [str(v) for v in unique_vals]
                print(
                    f"  ✓ Extracted {len(unique_vals)} unique values for '{attr_name}'"
                )
            else:
                print(
                    f"  ⚠ Skipped values for '{attr_name}' (too many: {len(unique_vals)})"
                )

        sdv_metadata["columns"][attr_name] = col_info

    # 🔍 DIAGNOSTIC: Show complete SDV metadata structure
    print(f"  Converted to SDV metadata with {len(sdv_metadata['columns'])} columns")
    # print("\n  📋 完整的 SDV Metadata Dictionary:")
    # print("  " + "=" * 70)
    # import json
    # print("  " + json.dumps(sdv_metadata, indent=4).replace("\n", "\n  "))
    # print("  " + "=" * 70)

    return sdv_metadata


def _map_attribute_to_sdv_type(attribute: Any, actual_dtype=None) -> str:
    """
    Map PETsARD Attribute to SDV sdtype 將 PETsARD Attribute 對應到 SDV sdtype

    Args:
        attribute: PETsARD Attribute object or dict PETsARD Attribute 物件或 dict
        actual_dtype: Actual pandas dtype from data (optional) 實際資料的 pandas dtype（可選）

    Returns:
        str: SDV sdtype
    """
    # Handle both dict and Attribute object 處理 dict 和 Attribute 物件兩種情況
    if isinstance(attribute, dict):
        # CRITICAL FIX: category 存在 type_attr 裡，不是 Attribute 的直接屬性
        category = attribute.get("type_attr", {}).get("category", False)
        logical_type = attribute.get("logical_type")
        attr_type = attribute.get("type")
        attr_name = attribute.get("name", "unknown")
    else:
        # CRITICAL FIX: category 存在 type_attr 裡，不是 Attribute 的直接屬性
        category = (
            attribute.type_attr.get("category", False) if attribute.type_attr else False
        )
        logical_type = attribute.logical_type
        attr_type = attribute.type
        attr_name = attribute.name

    # PRIORITY 0: Check actual data type first (most reliable)
    # 優先級 0：優先檢查實際資料類型（最可靠）
    # This handles cases where schema type is outdated after preprocessing
    # 這處理 schema 類型在預處理後過時的情況
    if actual_dtype is not None:
        import pandas as pd

        actual_dtype_str = str(actual_dtype)

        # If actual data is numerical, use numerical regardless of schema
        # 如果實際資料是數值類型，無論 schema 如何都使用 numerical
        if pd.api.types.is_numeric_dtype(actual_dtype):
            print(
                f"  ✓ '{attr_name}': actual_dtype={actual_dtype_str} → sdtype='numerical'"
            )
            return "numerical"
        elif pd.api.types.is_datetime64_any_dtype(actual_dtype):
            print(
                f"  ✓ '{attr_name}': actual_dtype={actual_dtype_str} → sdtype='datetime'"
            )
            return "datetime"
        elif pd.api.types.is_bool_dtype(actual_dtype):
            print(
                f"  ✓ '{attr_name}': actual_dtype={actual_dtype_str} → sdtype='boolean'"
            )
            return "boolean"
        # If actual data is object/string, continue to other checks
        # 如果實際資料是 object/string，繼續其他檢查

    # PRIORITY 0.5: Check category flag explicitly (before type checks)
    # 優先級 0.5：明確檢查 category 標記（在類型檢查之前）
    # This ensures category=True is respected even for numerical types
    # 這確保即使是數值類型，category=True 也會被尊重
    if category is True:
        print(f"  ✓ '{attr_name}': category=True → sdtype='categorical'")
        return "categorical"

    # PRIORITY 1: Check data type first for numerical types
    # 優先級 1：對於 numerical 類型，優先檢查資料類型
    # This prevents integer/float from being incorrectly marked as categorical
    # 這防止 integer/float 被錯誤地標記為 categorical
    if attr_type:
        attr_type_lower = attr_type.lower()
        if "int" in attr_type_lower or "float" in attr_type_lower:
            # For numerical types, only treat as categorical if category is explicitly True
            # AND it's a reasonable categorical variable (not high cardinality)
            # 對於 numerical 類型，只有當 category 明確為 True 時才視為 categorical
            # 注意：這裡我們信任 schema 的標記，但 SDV 會根據實際資料做最終判斷
            if category is True:
                # Integer/float marked as category: let SDV decide based on cardinality
                # 標記為 category 的 integer/float：讓 SDV 根據基數決定
                # For now, we'll trust the numerical type over the category flag
                # 目前，我們信任 numerical 類型而非 category 標記
                return "numerical"
            return "numerical"
        elif "bool" in attr_type_lower:
            return "boolean"
        elif "datetime" in attr_type_lower:
            return "datetime"

    # PRIORITY 2: Check logical_type for semantic types
    # 優先級 2：檢查 logical_type 的語義類型（email、phone 等）
    if logical_type:
        logical = logical_type.lower()
        if logical in ["email", "phone", "ssn", "credit_card"]:
            return "pii"
        elif logical in ["datetime", "date", "time", "timestamp"]:
            return "datetime"
        elif logical in ["address", "ip", "url"]:
            return "pii"

    # PRIORITY 3: Check category field when False (already handled True at 0.5)
    # 優先級 3：檢查 category=False 的情況（True 已在 0.5 處理）
    if category is False:
        # Explicitly marked as NOT categorical
        # 明確標記為非 categorical
        # For string types with category=false, check actual dtype or default to numerical
        # 對於 category=false 的字串類型，檢查實際 dtype 或預設為 numerical
        if attr_type:
            attr_type_lower = attr_type.lower()
            if attr_type_lower in ["string", "str", "object"]:
                # String type but category=false means it was likely encoded to numerical
                # 字串類型但 category=false 表示可能已編碼為數值
                # If we have actual_dtype and it's numeric, we would have returned at PRIORITY 0
                # So if we're here, use numerical as it's likely an encoded categorical
                # 如果有 actual_dtype 且為數值，在 PRIORITY 0 就會返回了
                # 所以到這裡，使用 numerical 因為可能是已編碼的類別
                print(
                    f"  ⚠️ '{attr_name}': schema says string but category=false → using 'numerical' (likely encoded)"
                )
                return "numerical"
        # For other types with category=false, continue to next checks
        # 其他 category=false 的類型，繼續下一個檢查

    # PRIORITY 4: Default based on remaining type
    # 優先級 4：根據剩餘類型設定預設值
    if attr_type:
        attr_type_lower = attr_type.lower()
        if attr_type_lower in ["string", "str", "object"]:
            # String types default to categorical (when category is not explicitly set)
            # 字串類型預設為 categorical（當 category 未明確設定時）
            return "categorical"

    # Default to categorical for unknown types
    # 未知類型預設為 categorical
    return "categorical"


class BaseSDVSynthesizer:
    """
    Base class for SDV Synthesizers SDV Synthesizer 基礎類別

    Base wrapper class for all SDV methods, handling common initialization
    and metadata conversion logic
    所有 SDV 方法的基礎封裝類，處理共同的初始化和元數據轉換邏輯
    """

    def __init__(self, config: dict, metadata: Schema = None):
        """
        Initialize SDV synthesizer 初始化 SDV synthesizer

        Args:
            config: Configuration dictionary with sample_num_rows etc. 配置字典，包含 sample_num_rows 等參數
            metadata: PETsARD Schema metadata (optional) PETsARD Schema 元數據（可選）
        """
        init_start = time.time()

        # Check if SDV is installed 檢查 SDV 是否安裝
        step_start = time.time()
        check_sdv_installed()

        from sdv.metadata import Metadata as SDV_Metadata

        self.config = config
        self.metadata = metadata
        self.logger = logging.getLogger(f"PETsARD.{self.__class__.__name__}")
        self.logger.info(f"⏱️  SDV import: {time.time() - step_start:.3f}s")

        # 🔍 DIAGNOSTIC: Check if metadata is provided
        print("=" * 70)
        print("🔍 DIAGNOSTIC: Checking metadata in __init__")
        print("=" * 70)
        if metadata is None:
            print("⚠️  WARNING: metadata is None! SDV will infer types from data later.")
        else:
            print(f"✓ Received metadata object: {type(metadata)}")
            if hasattr(metadata, "attributes"):
                print(f"✓ Metadata has {len(metadata.attributes)} attributes:")
                for attr_name, attr in list(metadata.attributes.items())[:5]:
                    print(
                        f"  - {attr_name}: type={getattr(attr, 'type', 'N/A')}, logical_type={getattr(attr, 'logical_type', 'N/A')}"
                    )
                if len(metadata.attributes) > 5:
                    print(f"  ... and {len(metadata.attributes) - 5} more attributes")
        print("=" * 70)

        self._synthesizer = None
        self._sdv_metadata = None
        self._original_dtypes = {}  # Record original dtypes for restoration
        self._original_pandas_dtypes = {}  # Record actual pandas dtypes from data

        # Handle device configuration 處理設備配置
        step_start = time.time()
        self._setup_device()
        self.logger.info(f"⏱️  Device setup: {time.time() - step_start:.3f}s")

        # If metadata is provided, convert to SDV format 如果提供了 metadata，預先轉換為 SDV 格式
        if metadata is not None:
            step_start = time.time()
            self.logger.info(
                f"Converting PETsARD Schema to SDV metadata (columns: {len(metadata.attributes)})"
            )
            print("🔍 DIAGNOSTIC: Converting metadata to SDV format...")
            sdv_metadata_dict = schema_to_sdv(metadata)
            self.logger.info(f"⏱️  Schema conversion: {time.time() - step_start:.3f}s")

            step_start = time.time()
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings(
                    "always", message="No table name was provided.*"
                )
                self._sdv_metadata = SDV_Metadata.load_from_dict(sdv_metadata_dict)
                for warning in w:
                    self.logger.debug(f"SDV Metadata: {warning.message}")
            print("✓ Successfully created SDV_Metadata object")
            print(f"  SDV_Metadata type: {type(self._sdv_metadata)}")
            self.logger.info(
                f"⏱️  SDV Metadata.load_from_dict: {time.time() - step_start:.3f}s"
            )

        self.logger.info(f"✅ Total __init__ time: {time.time() - init_start:.3f}s")

    def _setup_device(self):
        """
        Setup compute device (CPU/CUDA) 設定計算設備（CPU/CUDA）

        Raises:
            RuntimeError: If CUDA is requested but not available 如果要求 CUDA 但不可用
        """
        import torch

        # Get device setting from config (default: 'cpu') 從配置取得設備設定（預設：'cpu'）
        device_config = self.config.get("device", "cpu").lower()

        if device_config == "cuda":
            # User explicitly requested CUDA 用戶明確要求 CUDA
            if not torch.cuda.is_available():
                error_msg = (
                    "CUDA device requested but CUDA is not available! "
                    "CUDA 設備被要求但 CUDA 不可用！\n"
                    "Please check: 請檢查：\n"
                    "1. PyTorch is installed with CUDA support PyTorch 是否安裝了 CUDA 支援\n"
                    "2. NVIDIA GPU is available NVIDIA GPU 是否可用\n"
                    "3. CUDA drivers are properly installed CUDA 驅動是否正確安裝"
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)

            self.use_cuda = True
            self.logger.info("Using CUDA device for training 使用 CUDA 設備進行訓練")

        elif device_config == "cpu":
            # User requested CPU or default CPU 用戶要求 CPU 或預設 CPU
            self.use_cuda = False
            self.logger.info("Using CPU device for training 使用 CPU 設備進行訓練")

        else:
            # Invalid device specification 無效的設備規格
            error_msg = (
                f"Invalid device '{device_config}'. Must be 'cpu' or 'cuda'. "
                f"無效的設備 '{device_config}'。必須是 'cpu' 或 'cuda'。"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _create_sdv_metadata(self, data: pd.DataFrame):
        """
        Create SDV metadata from data (used when metadata not provided)
        從數據創建 SDV metadata（當未提供 metadata 時使用）

        Args:
            data: Training data 訓練數據
        """
        from sdv.metadata import Metadata as SDV_Metadata

        from petsard.metadater.metadater import SchemaMetadater

        print("\n🔍 DIAGNOSTIC: _create_sdv_metadata() called - FALLBACK MODE!")
        print(f"  Data shape: {data.shape}")
        print(f"  Data dtypes sample: {dict(list(data.dtypes.items())[:3])}")

        step_start = time.time()
        self.logger.info(f"Creating schema from data (shape: {data.shape})...")
        schema = SchemaMetadater.from_data(data)
        self.logger.info(
            f"  ⏱️  SchemaMetadater.from_data: {time.time() - step_start:.3f}s"
        )

        step_start = time.time()
        print("  Converting inferred schema to SDV format with data...")
        sdv_metadata_dict = schema_to_sdv(schema, data=data)
        self.logger.info(f"  ⏱️  schema_to_sdv: {time.time() - step_start:.3f}s")

        step_start = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always", message="No table name was provided.*")
            self._sdv_metadata = SDV_Metadata.load_from_dict(sdv_metadata_dict)
            for warning in w:
                self.logger.debug(f"SDV Metadata: {warning.message}")
        self.logger.info(
            f"  ⏱️  SDV Metadata.load_from_dict: {time.time() - step_start:.3f}s"
        )

    def _initialize_synthesizer(self):
        """
        Initialize SDV synthesizer instance (implemented by subclass)
        初始化 SDV synthesizer 實例（由子類實現）

        Raises:
            NotImplementedError: Must be implemented by subclass 必須由子類實現
        """
        raise NotImplementedError("Subclass must implement _initialize_synthesizer")

    def fit(self, data: pd.DataFrame) -> None:
        """
        Train the synthesizer 訓練 synthesizer

        Args:
            data: Training data 訓練數據
        """
        fit_start = time.time()
        self.logger.info(f"🚀 Starting fit() for {self.__class__.__name__}")

        # Record original dtypes from metadata if available, otherwise from data
        # This is crucial because data may have been preprocessed (e.g., Int64 -> float64)
        # 從 metadata 記錄原始 dtypes（如果可用），否則從數據記錄

        # TIMING: Metadata check
        step_start = time.time()
        self.logger.debug(f"[DEBUG] metadata is None: {self.metadata is None}")
        if self.metadata is not None:
            self.logger.debug(f"[DEBUG] metadata type: {type(self.metadata)}")
            self.logger.debug(
                f"[DEBUG] metadata has attributes: {hasattr(self.metadata, 'attributes')}"
            )
            if hasattr(self.metadata, "attributes"):
                self.logger.debug(
                    f"[DEBUG] attributes count: {len(self.metadata.attributes)}"
                )
        self.logger.info(f"⏱️  Metadata check: {time.time() - step_start:.3f}s")

        # TIMING: Extract original dtypes
        step_start = time.time()
        # 這很重要，因為資料可能已被預處理（例如 Int64 -> float64）
        if self.metadata is not None and hasattr(self.metadata, "attributes"):
            # Extract original dtypes from PETsARD Schema
            # 從 PETsARD Schema 提取原始 dtypes
            self._original_dtypes = {}
            for attr_name, attribute in self.metadata.attributes.items():
                if attr_name in data.columns:
                    # Use the type from metadata, which preserves original dtype
                    # 使用 metadata 中的類型，保留原始 dtype
                    original_type = attribute.type
                    self._original_dtypes[attr_name] = original_type
            self.logger.info(
                f"✓ Recorded {len(self._original_dtypes)} dtypes from metadata"
            )
        else:
            # Fallback: record dtypes from training data
            # 回退方案：從訓練數據記錄 dtypes
            self._original_dtypes = {col: data[col].dtype for col in data.columns}
            self.logger.warning(
                f"✗ Fallback - Recorded {len(self._original_dtypes)} dtypes from data"
            )
        self.logger.info(f"⏱️  Extract dtypes: {time.time() - step_start:.3f}s")

        # TIMING: Create SDV metadata
        step_start = time.time()
        # CRITICAL FIX: Always regenerate SDV metadata using actual data dtypes
        # 關鍵修復：始終使用實際數據的 dtype 重新生成 SDV metadata
        print("\n🔍 DIAGNOSTIC: Regenerating SDV metadata with actual data dtypes")

        if self.metadata is not None and hasattr(self.metadata, "attributes"):
            # We have PETsARD schema - use it with actual data to get correct types
            # 有 PETsARD schema - 結合實際數據來取得正確的類型
            print("✓ Using PETsARD schema + actual data dtypes")
            print(f"  Converting {len(self.metadata.attributes)} attributes...")

            # Convert schema to SDV format, passing data to check actual dtypes
            # 將 schema 轉換為 SDV 格式，傳入 data 以檢查實際 dtype
            sdv_metadata_dict = schema_to_sdv(self.metadata, data=data)

            import warnings

            from sdv.metadata import Metadata as SDV_Metadata

            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings(
                    "always", message="No table name was provided.*"
                )
                self._sdv_metadata = SDV_Metadata.load_from_dict(sdv_metadata_dict)
                for warning in w:
                    self.logger.debug(f"SDV Metadata: {warning.message}")

            print(
                f"✓ Successfully regenerated SDV metadata with {len(sdv_metadata_dict['columns'])} columns"
            )
        elif self._sdv_metadata is None:
            # No schema provided - create from data (fallback)
            # 沒有提供 schema - 從數據創建（備用方案）
            print("⚠️  No PETsARD schema! Will infer from data...")
            self.logger.warning(
                "⚠️  No PETsARD schema provided! Inferring types from data"
            )
            self._create_sdv_metadata(data)
        else:
            # Edge case: somehow have _sdv_metadata but no self.metadata
            # This shouldn't happen in normal flow
            print("⚠️  Have _sdv_metadata but no self.metadata - unusual case")
            print("   Keeping existing _sdv_metadata")

        self.logger.info(
            f"⏱️  Create/regenerate SDV metadata: {time.time() - step_start:.3f}s"
        )

        # TIMING: Initialize synthesizer
        step_start = time.time()
        # Initialize synthesizer 初始化 synthesizer
        if self._synthesizer is None:
            self.logger.info("Initializing synthesizer...")
            self._initialize_synthesizer()
        self.logger.info(f"⏱️  Initialize synthesizer: {time.time() - step_start:.3f}s")

        # TIMING: Train
        step_start = time.time()
        # Train 訓練
        self.logger.info(
            f"Training {self.__class__.__name__} with data shape: {data.shape}"
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._synthesizer.fit(data)
            for warning in w:
                self.logger.debug(f"Warning during fit: {warning.message}")

        self.logger.info(f"⏱️  SDV fit() execution: {time.time() - step_start:.3f}s")
        self.logger.info(f"✅ Total fit() time: {time.time() - fit_start:.3f}s")

    def sample(self) -> pd.DataFrame:
        """
        Generate synthetic data 生成合成數據

        Returns:
            pd.DataFrame: Synthetic data 合成數據
        """
        if self._synthesizer is None:
            raise RuntimeError("Synthesizer has not been fitted yet. Call fit() first.")

        num_rows = self.config.get("sample_num_rows", 100)
        batch_size = self.config.get("batch_size")

        self.logger.info(f"Sampling {num_rows} rows")

        synthetic_data = self._synthesizer.sample(
            num_rows=num_rows, batch_size=batch_size
        )

        # Restore original dtypes (especially important for Int64, Int32, etc.)
        # 恢復原始 dtypes（對 Int64, Int32 等特別重要）
        synthetic_data = self._restore_dtypes(synthetic_data)

        self.logger.info(f"Successfully generated {len(synthetic_data)} rows")
        return synthetic_data

    def _restore_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Restore original dtypes to synthetic data
        恢復合成數據的原始 dtype

        Args:
            data: Synthetic data 合成數據

        Returns:
            pd.DataFrame: Data with restored dtypes 恢復 dtype 後的數據
        """
        for col in data.columns:
            if col not in self._original_dtypes:
                continue

            original_dtype = self._original_dtypes[col]
            # Handle both pandas dtype objects and string representations
            # 處理 pandas dtype 物件和字串表示
            dtype_str = str(original_dtype)

            try:
                # Handle schema 'integer' type - map to Int64 (pandas nullable integer)
                # 處理 schema 的 'integer' 類型 - 映射到 Int64（pandas nullable integer）
                if dtype_str == "integer":
                    data[col] = data[col].round().astype("Int64")
                    self.logger.info(
                        f"✓ Restored {col} from schema type 'integer' to 'Int64'"
                    )
                # Handle Pandas nullable integer types 處理 Pandas nullable integer 類型
                elif dtype_str in [
                    "Int8",
                    "Int16",
                    "Int32",
                    "Int64",
                    "UInt8",
                    "UInt16",
                    "UInt32",
                    "UInt64",
                ]:
                    # Round to integer and convert to nullable integer type
                    # 四捨五入為整數並轉換為 nullable integer 類型
                    data[col] = data[col].round().astype(dtype_str)
                    self.logger.debug(
                        f"Restored {col} to nullable integer type {dtype_str}"
                    )
                # Handle standard integer types 處理標準整數類型
                elif (
                    any(
                        int_type in dtype_str.lower()
                        for int_type in [
                            "int8",
                            "int16",
                            "int32",
                            "int64",
                            "uint8",
                            "uint16",
                            "uint32",
                            "uint64",
                        ]
                    )
                    and "Int" not in dtype_str
                ):
                    # Standard numpy integer types
                    # 標準 numpy 整數類型
                    data[col] = data[col].round().astype(dtype_str)
                    self.logger.debug(
                        f"Restored {col} to standard integer type {dtype_str}"
                    )
                # Handle datetime types 處理日期時間類型
                elif "datetime" in dtype_str.lower():
                    if pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = pd.to_datetime(
                            data[col].round().astype("int64"), unit="ns"
                        )
                    else:
                        data[col] = pd.to_datetime(data[col])
                    self.logger.debug(f"Restored {col} to datetime type")
                # Handle timedelta types 處理時間差類型
                elif "timedelta" in dtype_str.lower():
                    if pd.api.types.is_numeric_dtype(data[col]):
                        data[col] = pd.to_timedelta(
                            data[col].round().astype("int64"), unit="ns"
                        )
                    else:
                        data[col] = pd.to_timedelta(data[col])
                    self.logger.debug(f"Restored {col} to timedelta type")
                # Handle float types 處理浮點數類型
                elif any(
                    float_type in dtype_str.lower()
                    for float_type in ["float16", "float32", "float64"]
                ):
                    data[col] = data[col].astype(dtype_str)
                    self.logger.debug(f"Restored {col} to float type {dtype_str}")
                # Handle boolean types 處理布林類型
                elif "bool" in dtype_str.lower():
                    data[col] = data[col].astype(dtype_str)
                    self.logger.debug(f"Restored {col} to boolean type {dtype_str}")
                # Handle string/object types 處理字串/物件類型
                elif dtype_str in ["object", "string", "str"]:
                    data[col] = data[col].astype("object")
                    self.logger.debug(f"Restored {col} to object type")
                # Try direct conversion for other types 其他類型嘗試直接轉換
                else:
                    data[col] = data[col].astype(dtype_str)
                    self.logger.debug(f"Restored {col} to type {dtype_str}")
            except Exception as e:
                self.logger.warning(
                    f"Failed to restore dtype for {col} to {dtype_str}: {e}, keeping as is"
                )

        return data


class SDV_GaussianCopula(BaseSDVSynthesizer):
    """
    SDV GaussianCopula Synthesizer

    Fast statistical method suitable for 基於統計分佈的快速合成方法，適合：
    - Fast prototyping 快速原型測試
    - Large datasets 大型資料集
    - Numerical data 數值型資料

    Supported parameters 支援參數：
        - default_distribution: Default distribution type 預設分佈類型 (truncnorm, beta, gamma, uniform, etc.)
        - numerical_distributions: Column-specific distributions 特定欄位的分佈設定
        - enforce_min_max_values: Enforce numerical range constraints 強制數值範圍限制
        - enforce_rounding: Force rounding for integer columns 強制整數欄位四捨五入
    """

    def _initialize_synthesizer(self):
        """Initialize GaussianCopula synthesizer 初始化 GaussianCopula synthesizer"""
        from sdv.single_table import GaussianCopulaSynthesizer

        init_params = {
            "metadata": self._sdv_metadata,
            "enforce_rounding": self.config.get("enforce_rounding", True),
            "enforce_min_max_values": self.config.get("enforce_min_max_values", True),
        }

        # Note: GaussianCopula doesn't use GPU, so cuda parameter is not applicable
        # 注意：GaussianCopula 不使用 GPU，因此 cuda 參數不適用
        if self.use_cuda:
            self.logger.warning(
                "GaussianCopula does not support CUDA acceleration. Using CPU. "
                "GaussianCopula 不支援 CUDA 加速。使用 CPU。"
            )

        # Add distribution-related parameters 添加分佈相關參數
        if "default_distribution" in self.config:
            init_params["default_distribution"] = self.config["default_distribution"]

        if "numerical_distributions" in self.config:
            init_params["numerical_distributions"] = self.config[
                "numerical_distributions"
            ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._synthesizer = GaussianCopulaSynthesizer(**init_params)
            for warning in w:
                self.logger.debug(f"Warning during initialization: {warning.message}")


class SDV_CTGAN(BaseSDVSynthesizer):
    """
    SDV CTGAN Synthesizer

    GAN-based deep learning method suitable for 基於 GAN 的深度學習方法，適合：
    - High-quality synthetic data 需要高品質合成資料
    - Complex data patterns 複雜資料模式
    - GPU-accelerated environments 有 GPU 加速的環境

    Supported parameters 支援參數：
        - epochs: Training epochs 訓練輪數 (default 預設 300)
        - batch_size: Batch size 批次大小 (default 預設 500, must be divisible by pac 必須能被 pac 整除)
        - pac: PAC (Packing) size PAC 打包大小 (default 預設 10, batch_size must be divisible by this)
        - discriminator_steps: Discriminator training steps 判別器訓練步數 (default 預設 1)
        - generator_lr: Generator learning rate 生成器學習率 (default 預設 0.0002)
        - discriminator_lr: Discriminator learning rate 判別器學習率 (default 預設 0.0002)
        - generator_dim: Generator dimensions 生成器維度 (default 預設 (256, 256))
        - discriminator_dim: Discriminator dimensions 判別器維度 (default 預設 (256, 256))
        - enforce_rounding: Force rounding for integer columns 強制整數欄位四捨五入
    """

    def _initialize_synthesizer(self):
        """Initialize CTGAN synthesizer 初始化 CTGAN synthesizer"""
        step_start = time.time()
        from sdv.single_table import CTGANSynthesizer

        self.logger.info(
            f"  ⏱️  Import CTGANSynthesizer: {time.time() - step_start:.3f}s"
        )

        init_params = {
            "metadata": self._sdv_metadata,
            "enforce_rounding": self.config.get("enforce_rounding", True),
            "epochs": self.config.get("epochs", 300),
            "batch_size": self.config.get("batch_size", 500),
            "cuda": self.use_cuda,  # Enable/disable CUDA 啟用/禁用 CUDA
        }

        # Add optional parameters 添加可選參數
        optional_params = [
            "pac",
            "discriminator_steps",
            "generator_lr",
            "discriminator_lr",
            "generator_dim",
            "discriminator_dim",
            "verbose",
        ]

        for param in optional_params:
            if param in self.config:
                init_params[param] = self.config[param]

        self.logger.info(
            f"Creating CTGANSynthesizer with params: epochs={init_params['epochs']}, batch_size={init_params['batch_size']}, cuda={init_params['cuda']}"
        )
        step_start = time.time()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._synthesizer = CTGANSynthesizer(**init_params)
            for warning in w:
                self.logger.debug(f"Warning during initialization: {warning.message}")
        self.logger.info(
            f"  ⏱️  CTGANSynthesizer() instantiation: {time.time() - step_start:.3f}s"
        )


class SDV_CopulaGAN(BaseSDVSynthesizer):
    """
    SDV CopulaGAN Synthesizer

    Combines Copula statistics and GAN, suitable for 結合 Copula 統計方法和 GAN，適合：
    - Mixed data types (continuous + discrete) 混合型資料（連續+離散）
    - Better marginal distribution modeling 需要更好的邊際分佈模擬
    - Balancing statistics and deep learning 平衡統計和深度學習優點

    Supported parameters 支援參數：
        - epochs: Training epochs 訓練輪數 (default 預設 300)
        - batch_size: Batch size 批次大小 (default 預設 500, must be divisible by pac 必須能被 pac 整除)
        - pac: PAC (Packing) size PAC 打包大小 (default 預設 10, batch_size must be divisible by this)
        - discriminator_steps: Discriminator training steps 判別器訓練步數 (default 預設 1)
        - generator_lr: Generator learning rate 生成器學習率 (default 預設 0.0002)
        - discriminator_lr: Discriminator learning rate 判別器學習率 (default 預設 0.0002)
        - default_distribution: Default distribution type 預設分佈類型 (default 預設 beta)
        - numerical_distributions: Column-specific distributions 特定欄位的分佈設定
        - enforce_rounding: Force rounding for integer columns 強制整數欄位四捨五入
    """

    def _initialize_synthesizer(self):
        """Initialize CopulaGAN synthesizer 初始化 CopulaGAN synthesizer"""
        from sdv.single_table import CopulaGANSynthesizer

        init_params = {
            "metadata": self._sdv_metadata,
            "enforce_rounding": self.config.get("enforce_rounding", True),
            "epochs": self.config.get("epochs", 300),
            "batch_size": self.config.get("batch_size", 500),
            "cuda": self.use_cuda,  # Enable/disable CUDA 啟用/禁用 CUDA
        }

        # Add distribution-related parameters 添加分佈相關參數
        if "default_distribution" in self.config:
            init_params["default_distribution"] = self.config["default_distribution"]

        if "numerical_distributions" in self.config:
            init_params["numerical_distributions"] = self.config[
                "numerical_distributions"
            ]

        # Add other optional parameters 添加其他可選參數
        optional_params = [
            "pac",
            "discriminator_steps",
            "generator_lr",
            "discriminator_lr",
            "generator_dim",
            "discriminator_dim",
            "verbose",
        ]

        for param in optional_params:
            if param in self.config:
                init_params[param] = self.config[param]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._synthesizer = CopulaGANSynthesizer(**init_params)
            for warning in w:
                self.logger.debug(f"Warning during initialization: {warning.message}")


class SDV_TVAE(BaseSDVSynthesizer):
    """
    SDV TVAE Synthesizer

    VAE-based generative model, suitable for 基於 VAE 的生成模型，適合：
    - Stable training process 需要穩定訓練過程
    - Medium-scale data 中等規模資料
    - Better convergence 較好的收斂性

    Supported parameters 支援參數：
        - epochs: Training epochs 訓練輪數 (default 預設 300)
        - batch_size: Batch size 批次大小 (default 預設 500)
        - encoder_layers: Encoder layer configuration 編碼器層設定 (default 預設 (128, 128))
        - decoder_layers: Decoder layer configuration 解碼器層設定 (default 預設 (128, 128))
        - compress_dims: Compression dimensions 壓縮維度 (default 預設 (128, 128))
        - decompress_dims: Decompression dimensions 解壓縮維度 (default 預設 (128, 128))
        - embedding_dim: Embedding dimension 嵌入維度 (default 預設 128)
        - l2scale: L2 regularization coefficient L2 正則化係數 (default 預設 1e-5)
        - loss_factor: Loss factor 損失因子 (default 預設 2)
        - enforce_min_max_values: Enforce numerical range constraints 強制數值範圍限制
        - enforce_rounding: Force rounding for integer columns 強制整數欄位四捨五入
    """

    def _initialize_synthesizer(self):
        """Initialize TVAE synthesizer 初始化 TVAE synthesizer"""
        from sdv.single_table import TVAESynthesizer

        init_params = {
            "metadata": self._sdv_metadata,
            "enforce_rounding": self.config.get("enforce_rounding", True),
            "enforce_min_max_values": self.config.get("enforce_min_max_values", True),
            "epochs": self.config.get("epochs", 300),
            "batch_size": self.config.get("batch_size", 500),
            "cuda": self.use_cuda,  # Enable/disable CUDA 啟用/禁用 CUDA
        }

        # Add network architecture parameters 添加網絡結構參數
        if "encoder_layers" in self.config:
            # Convert list to tuple 將列表轉換為 tuple
            init_params["compress_dims"] = tuple(self.config["encoder_layers"])

        if "decoder_layers" in self.config:
            init_params["decompress_dims"] = tuple(self.config["decoder_layers"])

        # Add other optional parameters 添加其他可選參數
        optional_params = ["embedding_dim", "l2scale", "loss_factor", "verbose"]

        for param in optional_params:
            if param in self.config:
                init_params[param] = self.config[param]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self._synthesizer = TVAESynthesizer(**init_params)
            for warning in w:
                self.logger.debug(f"Warning during initialization: {warning.message}")


# Provide aliases for convenience 為了方便使用，提供一個別名對照
SDV_METHODS = {
    "gaussiancopula": SDV_GaussianCopula,
    "ctgan": SDV_CTGAN,
    "copulagan": SDV_CopulaGAN,
    "tvae": SDV_TVAE,
}


if __name__ == "__main__":
    # Simple test program 簡單的測試程式
    print("SDV Custom Methods for PETsARD")
    print("=" * 50)
    print("\nAvailable Methods 可用的方法：")
    for method_name, method_class in SDV_METHODS.items():
        print(f"  - {method_name}: {method_class.__name__}")
        print(f"    {method_class.__doc__.strip().split('suitable for')[0].strip()}")

    print("\n" + "=" * 50)
    print("Usage 使用方式：")
    print("1. Set in YAML: method: custom_method 在 YAML 中設定 method: custom_method")
    print(
        "2. Specify: module_path: sdv-custom-methods.py 指定 module_path: sdv-custom-methods.py"
    )
    print(
        "3. Specify: class_name: SDV_GaussianCopula (or other methods) 指定 class_name: SDV_GaussianCopula (或其他方法)"
    )
    print(
        "\nFor complete examples, see sdv-custom-methods.yaml 請參考 sdv-custom-methods.yaml 獲取完整範例"
    )
