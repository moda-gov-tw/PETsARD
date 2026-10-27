import pandas as pd

from petsard.exceptions import UnableToLoadError
from petsard.loader.loader_base import LoaderBase


class LoaderPandasCsv(LoaderBase):
    """
    LoaderPandasCsv
        pandas.read_csv implementing of Loader
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): The configuration for the loader modules.

        Attr:
            config (dict): The configuration for the loader modules.
        """
        super().__init__(config)

    def load(self) -> pd.DataFrame:
        """
        Load and return the data

        Return:
            (pd.DataFrame)
                Data in csv by pd.DataFrame format.
        """
        # 1. set the filepath as first positional argument
        filepath = self.config["filepath"]

        pandas_config = {}

        # 2. If header_names is not None, setting custom header names
        if self.config.get("header_names") is not None:
            pandas_config.update({"header": 0, "names": self.config["header_names"]})
        else:
            # Default header settings to match original behavior
            pandas_config.update({"header": "infer", "names": None})

        # 3. assign dtype, na_values, and nrows
        list_setting = ["dtype", "na_values", "nrows"]
        pandas_config.update(
            {k: self.config[k] for k in list_setting if k in self.config}
        )

        try:
            return pd.read_csv(filepath, **pandas_config)
        except Exception as e:
            raise UnableToLoadError(
                f"Failed to load CSV file: {filepath}",
                filepath=filepath
            ) from e


class LoaderPandasExcel(LoaderBase):
    """
    LoaderPandasExcel
        pandas.read_excel implementing of Loader
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): The configuration for the loader modules.

        Attr:
            config (dict): The configuration for the loader modules.
        """
        super().__init__(config)

    def load(self) -> pd.DataFrame:
        """
        Load and return the data

        Return:
            (pd.DataFrame)
                Data in excel by pd.DataFrame format.
        """
        # Check if openpyxl is installed
        try:
            import openpyxl  # noqa: F401
        except ImportError as e:
            from petsard.exceptions import ConfigError

            raise ConfigError(
                "openpyxl is required to read Excel files. "
                "Please install it with: pip install petsard[xlsx]"
            ) from e

        # 1. set the filepath as first positional argument
        filepath = self.config["filepath"]

        pandas_config = {}

        # 2. If header_names is not None, setting custom header names
        if self.config.get("header_names") is not None:
            pandas_config.update({"header": 0, "names": self.config["header_names"]})
        else:
            # Default header settings to match original behavior
            pandas_config.update({"header": "infer", "names": None})

        # 3. assign dtype, na_values, and nrows
        list_setting = ["dtype", "na_values", "nrows"]
        pandas_config.update(
            {k: self.config[k] for k in list_setting if k in self.config}
        )

        try:
            return pd.read_excel(filepath, **pandas_config)
        except Exception as e:
            raise UnableToLoadError(
                f"Failed to load Excel file: {filepath}",
                filepath=filepath
            ) from e


class LoaderPandasJson(LoaderBase):
    """
    LoaderPandasJson
        pandas.read_json implementation of Loader
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): The configuration for the loader modules.

        Attr:
            config (dict): The configuration for the loader modules.
        """
        super().__init__(config)

    def load(self) -> pd.DataFrame:
        """
        Load and return the data

        Return:
            (pd.DataFrame)
                Data from JSON in pd.DataFrame format.
        """
        # 1. Set the filepath from the configuration
        filepath = self.config["filepath"]

        pandas_config = {}

        # 2. Assign JSON-specific settings (e.g., orient, lines, dtype, convert_dates)
        # 'orient' defines the expected JSON string format (e.g., 'records', 'columns')
        # 'lines' is set to True to read the file as a JSON object per line
        list_setting = ["orient", "lines", "dtype", "convert_dates"]
        pandas_config.update(
            {k: self.config[k] for k in list_setting if k in self.config}
        )

        try:
            # 3. Load the JSON file into a DataFrame
            return pd.read_json(filepath, **pandas_config)
        except Exception as e:
            raise UnableToLoadError(
                f"Failed to load JSON file: {filepath}",
                filepath=filepath
            ) from e
