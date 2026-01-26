"""
Data scaling utilities for preprocessing.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Scaler:
    """Unified scaler interface supporting multiple scaling methods."""

    def __init__(self, method: str = "standardize") -> None:
        """
        Initialize the scaler.

        Parameters
        ----------
        method : str
            Scaling method. One of: "standardize", "normalize", "none".
        """
        self.method = method.lower()
        if self.method not in ("standardize", "normalize", "none"):
            raise ValueError(
                f"method must be one of: standardize, normalize, none. Got: {method}"
            )

        self._scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None

    def fit(self, data: Union[np.ndarray, pd.DataFrame]) -> "Scaler":
        """
        Fit the scaler on data.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Input data to fit the scaler on.

        Returns
        -------
        self
        """
        if self.method == "standardize":
            self._scaler = StandardScaler()
            self._scaler.fit(data)
        elif self.method == "normalize":
            self._scaler = MinMaxScaler()
            self._scaler.fit(data)
        else:  # "none"
            self._scaler = None

        return self

    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using the fitted scaler.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Input data to transform.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        if self.method == "none":
            return np.asarray(data)

        if self._scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        transformed_data = self._scaler.transform(data)
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(transformed_data, columns=data.columns, index=data.index)
        return transformed_data

    def fit_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit and transform data in one step.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Input data to fit and transform.

        Returns
        -------
        np.ndarray
            Transformed data.
        """
        self.fit(data)
        return self.transform(data)
