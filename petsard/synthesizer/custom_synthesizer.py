import logging
import sys
from pathlib import PureWindowsPath

import pandas as pd

from petsard.exceptions import ConfigError, UnableToLoadError
from petsard.metadater import Schema
from petsard.synthesizer.synthesizer_base import BaseSynthesizer
from petsard.utils import load_external_module

# Module files of the initial methods removed from this version and the replacement
# named in the error message when such a module_path is not found
_ADVGAN_HINT = (
    "The initial AdvGAN synthesizer has been removed from this version; "
    "use method: DPflow_AdvGAN (see the Differentially Private Synthesizers page)."
)
REMOVED_INITIAL_MODULES: dict[str, str] = {
    "LLM_synthesizer.py": (
        "The initial LLM synthesizer has been removed from this version; use "
        "module_path: petsard/synthesizer/DP_LLM_synthesizer.py with "
        "class_name: DP_LLM_synthesizer (see the Differentially Private Synthesizers page)."
    ),
    "AdvGAN.py": _ADVGAN_HINT,
    "DP_AdvGAN.py": _ADVGAN_HINT,
    "DP_GAN.py": _ADVGAN_HINT,
}


class CustomSynthesizer(BaseSynthesizer):
    """
    Adapter class for Custom synthesizer as method = 'custom_method'
    """

    REQUIRED_METHODS: dict[str, str] = {
        "__init__": ["config", "metadata"],
        "fit": ["data"],
        "sample": [],
    }

    def __init__(self, config: dict, metadata: Schema = None):
        """
        Args:
            config (dict): The configuration assign by Synthesizer
            metadata (Schema, optional): The metadata object.

        Attributes:
            _logger (logging.Logger): The logger object.
            config (dict): The configuration of the synthesizer_base.
            _impl (Any): The synthesizer object.
        """
        super().__init__(config, metadata)
        self._logger: logging.Logger = logging.getLogger(
            f"PETsARD.{self.__class__.__name__}"
        )

        if "module_path" not in self.config:
            error_msg: str = (
                "Module path (module_path) is not provided in the configuration."
            )
            self._logger.error(error_msg)
            raise ConfigError(error_msg)

        if "class_name" not in self.config:
            error_msg: str = (
                "Class name (class_name) is not provided in the configuration."
            )
            self._logger.error(error_msg)
            raise ConfigError(error_msg)

        synthesizer_class: callable = None

        # Pass sys.path as search_paths to support notebook environments
        # This allows finding modules in the same directory as the notebook
        try:
            _, synthesizer_class = load_external_module(
                module_path=self.config["module_path"],
                class_name=self.config["class_name"],
                logger=self._logger,
                required_methods=self.REQUIRED_METHODS,
                search_paths=sys.path,  # Include Python path for module search
            )
        except UnableToLoadError as e:
            # PureWindowsPath splits on both "/" and "\"
            removed_hint = REMOVED_INITIAL_MODULES.get(
                PureWindowsPath(str(self.config["module_path"])).name
            )
            if removed_hint is None:
                raise
            error_msg = f"{e.message}\n{removed_hint}"
            self._logger.error(removed_hint)
            raise UnableToLoadError(
                error_msg, filepath=self.config["module_path"]
            ) from e

        self._impl = synthesizer_class(config=config, metadata=metadata)

    def _fit(self, data: pd.DataFrame) -> None:
        """
        Fit the synthesizer.
            _impl should be initialized in this method.

        Args:
            data (pd.DataFrame): The data to be fitted.
        """
        self._impl.fit(data)

    def _sample(self) -> pd.DataFrame:
        """
        Sample from the fitted synthesizer.

        Return:
            (pd.DataFrame): The synthesized data.
        """
        return self._impl.sample()
