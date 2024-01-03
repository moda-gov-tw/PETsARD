from ..Processor.Encoder import *
from ..Processor.Missingist import *
from ..Processor.Outlierist import *
from ..Processor.Scaler import *
from .Mediator import *
from ..Error import *

from copy import deepcopy
import logging

logging.basicConfig(level = logging.DEBUG, filename = 'log.txt', filemode = 'w',
                    format = '[%(levelname).1s %(asctime)s] %(message)s',
                    datefmt = '%Y%m%d %H:%M:%S')

# TODO - finish get_changes

class HyperProcessor:

    # object datatype indicates the unusual data,
    # passive actions will be taken in processing procedure

    _DEFAULT_MISSINGIST = {'numerical': Missingist_Mean, 
                           'categorical': Missingist_Drop,
                           'datetime': Missingist_Drop,
                           'object': Missingist_Drop}
    
    _DEFAULT_OUTLIERIST = {'numerical': Outlierist_IQR,
                           'categorical': None,
                           'datatime': Outlierist_IQR,
                           'object': None}
    
    _DEFAULT_ENCODER = {'numerical': None,
                        'categorical': Encoder_Uniform,
                        'datetime': None,
                        'object': Encoder_Uniform}
    
    _DEFAULT_SCALER = {'numerical': Scaler_Standard,
                       'categorical': None,
                       'datetime': Scaler_Standard,
                       'object': None}
    
    _DEFAULT_SEQUENCE = ['missingist', 'outlierist', 'encoder', 'scaler']
    


    def __init__(self, metadata: dict, config: dict=None) -> None:
        self._check_metadata_valid(metadata=metadata)
        self._metadata = metadata
        self._infer_metadata_dtype(metadata=metadata)

        # processing sequence
        self._sequence = None
        self._fitting_sequence = None
        self._inverse_sequence = None
        self._is_fitted = False

        # deal with global transformation of missingist and outlierist
        self.mediator_missingist = None
        self.mediator_outlierist = None

        # global NA values imputation
        self._na_percentage_global = metadata['metadata_global'].get('na_percentage', 0.0)
        self.rng = np.random.default_rng()

        self._config = dict()

        if config is None:
            self._generate_config()
        else:
            self.set_config(config=config)

    def _check_metadata_valid(self, metadata: dict) -> None:
        """
        Check whether the metadata contains the proper keys (metadata_col and metadata_global) for generating config.

        Input:
            metadata (dict): Metadata from the class Metadata or with the same format.

        Output:
            None
        """
        # check the structure of metadata
        if type(metadata) != dict:
            raise TypeError('Metadata should be a dict.')
        
        if not ('metadata_col' in metadata and 'metadata_global' in metadata):
            raise ValueError("'metadata_col' and 'metadata_global' should be in the metadata.")
        
        if type(metadata['metadata_col']) != dict:
            raise TypeError('metadata_col should be a dict.')
        
        if type(metadata['metadata_global']) != dict:
            raise TypeError('metadata_global should be a dict.')
        
        for v in metadata['metadata_col'].values():
            if type(v) != dict:
                raise TypeError('The elements in metadata_col should be a dict.')

    def _check_config_valid(self, config_to_check: dict=None) -> None:
        """
        Check whether the config contains valid preprocessors. It checks the validity of column names, the validity of processor types (i.e., dict keys), and the validity of processor objects (i.e., dict values).

        Input:
            config (dict, default=None): Config generated by the class or with the same format.

        Output:
            None
        """
        if config_to_check is None:
            raise ValueError('A config should be passed.')

        # check the structure of config
        if type(config_to_check) != dict:
            raise TypeError('Config should be a dict.')

        # check the validity of processor types
        if not set(config_to_check.keys()).issubset({'missingist', 'outlierist', 'encoder', 'scaler'}):
            raise ValueError(f'Invalid config processor type in the input dict, please check the dict keys of processor types.')

        for processor, processor_class in {'missingist': Missingist, 'outlierist': Outlierist, 'encoder': Encoder, 'scaler': Scaler}.items():

            if config_to_check.get(processor, None) is None:
                continue
            
            if type(config_to_check[processor]) != dict:
                raise TypeError('The config in each processor should be a dict.')
            
            # check the validity of column names (keys)
            if not set(config_to_check[processor].keys()).issubset(set(self._metadata['metadata_col'].keys())):
                raise ValueError(f'Some columns in the input config {processor} are not in the metadata. Please check the config or metadata again.')

            for col in config_to_check[processor].keys():
                # check the validity of processor objects (values)
                obj = config_to_check[processor].get(col, None)

                if not(isinstance(obj, processor_class) or obj is None):
                    raise ValueError(f'{col} from {processor} contain(s) invalid processor object(s), please check them again.')
                
    def _infer_metadata_dtype(self, metadata: dict) -> str:
        """
        Infer data types from the metadata. Used for generating config.

        The infer data types can be one of the following: 'numerical', 'categorical', 'datetime', and 'object'.

        Input:
            metadata (dict): The metadata to be inferred.

        Output:
            None
        """
        for col, val in metadata['metadata_col'].items():
            dtype = val.get('dtype', None)
            if dtype is None:
                raise ValueError(f'{col} should have a valid type.')
            
            if pd.api.types.is_numeric_dtype(dtype):
                self._metadata['metadata_col'][col]['infer_dtype'] = 'numerical'
            elif isinstance(dtype, pd.CategoricalDtype):
                self._metadata['metadata_col'][col]['infer_dtype'] = 'categorical'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                self._metadata['metadata_col'][col]['infer_dtype'] = 'datetime'
            elif pd.api.types.is_object_dtype(dtype):
                self._metadata['metadata_col'][col]['infer_dtype'] = 'object'
            else:
                raise ValueError(f'Invalid data type for {col}')
                    
    def _generate_config(self) -> None:
        """
        Generate config based on the metadata.

        Config structure: {processor_type: {col_name: processor_obj}}

        Input:
            None: The metadata is stored in the instance itself.

        Output:
            None: The config will be stored in the instance itself.
        """
        self._config = None # initialise the dict
        self._config = {'missingist': {},
                        'outlierist': {},
                        'encoder': {},
                        'scaler': {}}

        for col, val in self._metadata['metadata_col'].items():

            processor_dict = {'missingist': self._DEFAULT_MISSINGIST[val['infer_dtype']]()\
                               if self._DEFAULT_MISSINGIST[val['infer_dtype']] is not None else None,
                            'outlierist': self._DEFAULT_OUTLIERIST[val['infer_dtype']]()\
                               if self._DEFAULT_OUTLIERIST[val['infer_dtype']] is not None else None,
                            'encoder': self._DEFAULT_ENCODER[val['infer_dtype']]()\
                               if self._DEFAULT_ENCODER[val['infer_dtype']] is not None else None,
                            'scaler': self._DEFAULT_SCALER[val['infer_dtype']]()\
                               if self._DEFAULT_SCALER[val['infer_dtype']] is not None else None}
            
            for processor, obj in processor_dict.items():
                self._config[processor][col] = obj


    def get_config(self, col: list=None, print_config: bool=False) -> dict:
        """
        Get the config from the instance.

        Input:
            col (list): The columns the user want to get the config from. If the list is empty, all columns from the metadata will be selected.
            print_config (bool, default=False): Whether the result should be printed.

        Output:
            (dict): The config with selected columns.
        """
        get_col_list = []
        result_dict = {'missingist': {},
                       'outlierist': {},
                       'encoder': {},
                       'scaler': {}}

        if col:
            get_col_list = col
        else:
            get_col_list = list(self._metadata['metadata_col'].keys())

        if print_config:
            for processor in self._config.keys():
                print(processor)
                for colname in get_col_list:
                    print(f'    {colname}: {self._config[processor][colname].__class__}')
                    result_dict[processor][colname] = self._config[processor][colname]
        else:
            for processor in self._config.keys():
                for colname in get_col_list:
                    result_dict[processor][colname] = self._config[processor][colname]


        return result_dict

    def set_config(self, config: dict) -> None:
        """
        Edit the whole config. To keep the structure of the config, it fills the unspecified preprocessors with None. To prevent from this, use update_config() instead.

        Input:
            config (dict): The dict with the same format as the config class.

        Output:
            None
        """
        self._check_config_valid(config_to_check=config)

        for processor, val in self._config.items():
            if processor not in config.keys():
                config[processor] = {}
            for col in val.keys():
                self._config[processor][col] = config[processor].get(col, None)

    def update_config(self, config: dict) -> None:
        """
        Update part of the config.

        Input:
            config (dict): The dict with the same format as the config class.

        Output:
            None
        """
        self._check_config_valid(config_to_check=config)

        for processor, val in config.items():
            for col, obj in val.items():
                self._config[processor][col] = obj

    def fit(self, data: pd.DataFrame, sequence: list=None) -> None:
        """
        Fit the data.

        Input:
            data (pd.DataFrame): The data to be fitted.
            sequence (list): The processing sequence. 
                Avaliable procedures: 'missingist', 'outlierist', 'encoder', 'scaler'.
                This is the default sequence.

        Output:
            None
        """

        if sequence is None:
            self._sequence = self._DEFAULT_SEQUENCE
        else:
            self._check_sequence_valid(sequence)
            self._sequence = sequence

        self._fitting_sequence = self._sequence.copy()

        if 'missingist' in self._sequence:
            # if missingist is in the procedure, Mediator_Missingist should be in the queue right after the missingist
            self.mediator_missingist = Mediator_Missingist(self._config)
            self._fitting_sequence.insert(self._fitting_sequence.index('missingist')+1, self.mediator_missingist)
            logging.info('Mediator_Missingist is created.')

        if 'outlierist' in self._sequence:
            # if outlierist is in the procedure, Mediator_Outlierist should be in the queue right after the outlierist
            self.mediator_outlierist = Mediator_Outlierist(self._config)
            self._fitting_sequence.insert(self._fitting_sequence.index('outlierist')+1, self.mediator_outlierist)
            logging.info('Mediator_Outlierist is created.')

        self._detect_edit_global_transformation()

        for processor in self._fitting_sequence:
            if type(processor) == str:
                for col, obj in self._config[processor].items():

                    logging.debug(f'{processor}: {obj} from {col} start processing.')

                    if obj is None:
                        continue
                    
                    obj.fit(data[col])

                logging.info(f'{processor} fitting done.')
            else:
                # if the processor is not a string,
                # it should be a mediator, which could be fitted directly.

                logging.debug(f'mediator: {processor} start processing.')
                processor.fit(data)
                logging.info(f'{processor} fitting done.')
        

        self._is_fitted = True

    def _check_sequence_valid(self, sequence: list) -> None:
        """
        Check whether the sequence is valid.

        Input:
            sequence (list): The processing sequence.

        Output:
            None
        """
        if type(sequence) != list:
            raise TypeError('Sequence should be a list.')
        
        if len(sequence) == 0:
            raise ValueError('There should be at least one procedure in the sequence.')
        
        if len(sequence) > 4:
            raise ValueError('Too many procedures!')
        
        if len(list(set(sequence))) != len(sequence):
            raise ValueError('There are duplicated procedures in the sequence, please remove them.')
        
        for processor in sequence:
            if processor not in ['missingist', 'outlierist', 'encoder', 'scaler']:
                raise ValueError(f'{processor} is invalid, please check it again.')
            
    def _detect_edit_global_transformation(self) -> None:
        """
        Detect whether a processor in the config conducts global transformation.
        If it does, suppress other processors in the config by replacing them to the global one.
        Only works with Outlierist currently.

        Input:
            None

        Output:
            None
        """
        is_global_transformation = False
        replaced_class = None

        for obj in self._config['outlierist'].values():
            if obj is None:
                continue
            if obj.IS_GLOBAL_TRANSFORMATION:
                is_global_transformation = True
                replaced_class = obj.__class__
                logging.info(f'Global transformation detected. All processors will be replaced to {replaced_class}.')
                break

        if is_global_transformation:
            for col, obj in self._config['outlierist'].items():
                self._config['outlierist'][col] = replaced_class()

    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data through a series of procedures.

        Input:
            data (pd.DataFrame): The data to be transformed.

        Output:
            transformed (pd.DataFrame): The transformed data.
        """
        if not self._is_fitted:
            raise UnfittedError('The object is not fitted. Use .fit() first.')
        
        transformed = deepcopy(data)
        
        for processor in self._fitting_sequence:
            if type(processor) == str:
                for col, obj in self._config[processor].items():

                    logging.debug(f'{processor}: {obj} from {col} start transforming.')

                    if obj is None:
                        continue
                    
                    transformed[col] = obj.transform(transformed[col])

                logging.info(f'{processor} transformation done.')
            else:
                # if the processor is not a string,
                # it should be a mediator, which transforms the data directly.

                logging.debug(f'mediator: {processor} start transforming.')
                logging.debug(f'before transformation: data shape: {transformed.shape}')
                transformed = processor.transform(transformed)
                logging.debug(f'after transformation: data shape: {transformed.shape}')
                logging.info(f'{processor} transformation done.')

        return transformed

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data through a series of procedures.

        Input:
            data (pd.DataFrame): The data to be inverse transformed.

        Output:
            transformed (pd.DataFrame): The inverse transformed data.
        """
        if not self._is_fitted:
            raise UnfittedError('The object is not fitted. Use .fit() first.')
        
        # set NA percentage in Missingist
        index_list = list(self.rng.choice(data.index, 
                                     size=int(data.shape[0]*self._na_percentage_global), 
                                     replace=False).ravel())

        for col, obj in self._config['missingist'].items():
            if obj is None:
                continue
            obj.set_imputation_index(index_list)

            try:
                # the NA percentage taking global NA percentage into consideration
                adjusted_na_percentage = self._metadata['metadata_col'][col].get('na_percentage', 0.0)\
                    /self._na_percentage_global
            # if there is no NA in the original data
            except ZeroDivisionError:
                adjusted_na_percentage = 0.0
            
            obj.set_na_percentage(adjusted_na_percentage)
        
        # there is no method for restoring outliers
        self._inverse_sequence = self._sequence.copy()
        if 'outlierist' in self._inverse_sequence:
            self._inverse_sequence.remove('outlierist')
        
        transformed = deepcopy(data)
        
        # mediators are not involved in the inverse_transform process.
        for processor in self._inverse_sequence:
            for col, obj in self._config[processor].items():

                logging.debug(f'{processor}: {obj} from {col} start inverse transforming.')

                if obj is None:
                    continue
                
                transformed[col] = obj.inverse_transform(transformed[col])

            logging.info(f'{processor} inverse transformation done.')

        return transformed

    
    # determine whether the processors are not default settings
    def get_changes(self):
        pass

