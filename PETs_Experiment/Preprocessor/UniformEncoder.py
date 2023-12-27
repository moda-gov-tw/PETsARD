import numpy as np
import pandas as pd
from PETs_Experiment.Error import UnfittedError

"""
TODO - Categorical types are not accepted. Should correct this.
"""

class UniformEncoder:
    def __init__(self):
        # Mapping dict
        self.cat_to_val = None

        # Labels, lower and upper values
        self.labels = None
        self.upper_values = None
        self.lower_values = None

        self._is_fitted = False

        # Initiate a random generator
        self._rgenerator = np.random.default_rng()

    def fit(self, data):
        """
        Gather information for transformation and reverse transformation.

        Input:
            data (pd.Series): the categorical data needed to be transformed

        Output:
            None
        """
        normalize_value_counts = data.value_counts(normalize=True)
        # Get keys (original labels)
        self.labels = normalize_value_counts.index.get_level_values(0).to_list()
        # Get values (upper and lower bounds)
        self.upper_values = np.cumsum(normalize_value_counts.values)
        self.lower_values = np.roll(self.upper_values, 1)
        # To make sure the range of the data is in [0, 1]. That is, the range of an uniform dist.
        self.upper_values[-1] = 1.0
        self.lower_values[0] = 0.0

        self.cat_to_val = dict(zip(self.labels, list(zip(self.lower_values, self.upper_values))))

        self._is_fitted = True

    def transform(self, data):
        """
        Transform categorical data to a uniform distribution. For example, a column with two categories (e.g., 'Male', 'Female') can be mapped to [0.0, 0.5) and [0.5, 1], respectively.

        Input:
            data (pd.Series): the categorical data needed to be transformed

        Output:
            (pd.Series): the transformed data
        """

        # Check the object is fitted
        if not self._is_fitted:
            raise UnfittedError('The object is not fitted. Use .fit() first.')
        
        # Check whether the categories of the column are included in the fitted instance
        if not set(data.unique()).issubset(set(self.labels)): # type: ignore
            raise ValueError("The data contains categories that the object hasn't seen in the fitting process. Please check the data categories again.")
        
        if isinstance(data.dtype, pd.api.types.CategoricalDtype):
            data_obj = data.astype(object)
        
        return data_obj.map(lambda x: self._rgenerator.uniform(self.cat_to_val[x][0], self.cat_to_val[x][1], size=1)[0]) # type: ignore
    
    def inverse_transform(self, data):
        """
        Reverse the transformed data to the categorical data.

        Input:
            data (pd.Series): the categorical data needed to be transformed reversely

        Output:
            (pd.Series): the reverse transformed data
        """
        # Check the object is fitted
        if not self._is_fitted:
            raise UnfittedError('The object is not fitted. Use .fit() first.')
        
        # Check the range of the data is valid
        if data.max() > 1 or data.min() < 0:
            raise ValueError("The range of the data is out of range. Please check the data again.")
        
        bins_val = np.append(self.lower_values, 1.0)
        
        return pd.cut(data, right=False, include_lowest=True, bins=bins_val, labels=self.labels, ordered=False)
        


