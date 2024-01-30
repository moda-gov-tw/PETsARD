from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

from PETsARD.Error import ConfigError


@dataclass
class ProcessorConfig():
    """
    The Config dataclass of Processor
    ...

    colnames (List)
        The column names of data which will be processing

    config (Dict[sub-processor_name: None|str|List|dict])
        config should be any type of below:

        1. Not apply: Don't apply specific submodule.
            config = {
                'submodule_name': None
            }

        2. Simplest: One method in submodule apply to all columns.
            config = {
                'submodule_name': {
                    'method': 'method_name_1',
                    'all': True
                }
            }

        3. Single filter: One method in submodule apply to columns
            that only include/exclude certain fields.
            config = {
                'submodule_name': {
                    'method': 'method_name_1',
                    'include': 'colname_1'
                }
            }
            config = {
                'submodule_name': {
                    'method': 'method_name_1',
                    'exclude': 'colname_1'
                }
            }
            config = {
                'submodule_name': {
                    'method': 'method_name_1',
                    'include': ['colname_1','colname_2']
                }
            }

        4. Multiple methods: Record different methods in a list
            to apply them to different columns in the submodule.
            config = {
                'submodule_name': [
                    {'method': 'method_name_1',
                    'include': 'colname_1'
                    },
                    {'method': 'method_name_2',
                    'exclude': 'colname_1'
                    },
                ]
            }
    ...
    TODO verify method list
    TODO accept less method str (e.g. drop in missingist =auto=> missingist_drop)
    TODO support variable type input
    """
    colnames: List[str]
    config: Dict[
        str,
        Union[
            None,
            Dict[str, Union[str, Dict[str, str]]],
            List[Dict[str, Union[str, Dict[str, str]]]]
        ]
    ] = field(default_factory=dict)

    def __post_init__(self):
        SUBPROCESSOR = ['missingist', 'outlierist', 'encoder', 'scaler']
        config_transform = {}

        if not isinstance(self.colnames, list) \
                or not all(isinstance(col, str) for col in self.colnames):
            raise ConfigError(
                "ProcessorConfig: colnames should be a list of strings.")

        for proc_name, proc_config in self.config.items():
            config_transform[proc_name] = self._single_config(
                proc_name, proc_config)

        self.config_transform = config_transform

    def _single_config_dict(
        self,
        proc_name: str,
        proc_config: Dict[str, Union[str, Dict[str, str]]]
    ) -> Dict[str, str]:
        method = proc_config.setdefault('method', None)
        all = proc_config.setdefault('all',    False)
        include = proc_config.setdefault(
            'include', proc_config.setdefault('incl', None))
        exclude = proc_config.setdefault(
            'exclude', proc_config.setdefault('excl', None))

        if method is None or not isinstance(method, str):
            raise ConfigError(
                f"ProcessorConfig - {proc_name}: Invalid method value: \n"
                f"{method}."
            )

        if all:
            # Config 2nd. - Simplest: One method in submodule apply to all columns.
            return {colname: method for colname in self.colnames}

        if include and exclude:
            raise ConfigError(
                f"ProcessorConfig - {proc_name}: Invalid columns value.\n"
                f"Both include/exclude cannot be set simultaneously in 'columns'."
            )
        elif not (isinstance(include, (str, list)) or include is None)\
                or not (isinstance(exclude, (str, list)) or exclude is None):
            raise ConfigError(
                f"ProcessorConfig - {proc_name}: Invalid columns value.\n"
                f"include/exclude should be a str or lists."
            )

        # convert to list if they are not already
        include = [include] if isinstance(include, str) else (include or [])
        exclude = [exclude] if isinstance(exclude, str) else (exclude or [])
        invalid_columns = [item for item in include + exclude if item not in self.colnames]
        if invalid_columns:
            raise ConfigError(
                f"ProcessorConfig - {proc_name}: Invalid columns value.\n"
                f"The following columns do not belong to column names:\n"
                f"{', '.join(invalid_columns)}."
            )

        if len(include) >= 1:
            return {colname: method for colname in include}
        elif len(exclude) >= 1:
            return {colname: method for colname in self.colnames if colname not in exclude}

    def _single_config(
        self,
        proc_name: str,
        proc_config: Union[
            None,
            Dict[str, Union[str, Dict[str, str]]],
            List[Dict[str, Union[str, Dict[str, str]]]]
        ]
    ) -> Dict[str, str]:
        """
        Verify Sub-Processor config:
        1. The subconfig should be a dict
        2. The subconfig should have a key "method", and its value should be str
        3. The subconfig should have a key "columns", and its value should be
            string 'all', specific dictionary, or list of this specific dictionary
            3-a. The specific dictionary should have key(s) focus and/or ignore
            3-b. Their value(s) should be str or List[Str]
            3-c. Their value(s) should be included in colnames
            3-d. If it both contains focus and ignore, than their value should NOT overlay
        """
        if proc_config is None:
            # Config 1st. - Not apply: Don't apply specific submodule.
            return {colname: None for colname in self.colnames}
        elif isinstance(proc_config, dict):
            # Config 3rd. Single filter: One method in submodule apply to columns
            # that only include/exclude certain fields.
            return self._single_config_dict(proc_name, proc_config)
        elif isinstance(proc_config, list):
            # Config 4th. Multiple methods: Record different methods in a list
            # to apply them to different columns in the submodule.
            configs = [
                self._single_config_dict(proc_name, single_config) for single_config in proc_config
            ]
            # deduplicated
            combined_config = {}
            for single_config in configs:
                for colname, method in single_config.items():
                    if colname in combined_config:
                        raise ConfigError(
                            f"ProcessorConfig - {proc_name}: Invalid columns value.\n"
                            f"Duplicate column name {colname} found in the list of dictionaries."
                        )
                    else:
                        combined_config[colname] = method
            return combined_config
        else:
            raise ConfigError(
                f"ProcessorConfig - {proc_name}: Invalid columns value.\n"
                f"Only None, str, Dict, and List is allowed,\n"
                f"but it is currently {type(columns)} in '{columns}'."
            )
