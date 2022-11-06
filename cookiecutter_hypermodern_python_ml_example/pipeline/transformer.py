"""Transformer class for performing feature transformations on raw data."""

from collections.abc import Sequence
from typing import Any
from typing import Optional
from typing import Union

import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from cookiecutter_hypermodern_python_ml_example.contracts.contract_transformer import (
    ColumnTransformerContract,
)
from cookiecutter_hypermodern_python_ml_example.contracts.contract_transformer import (
    EstimatorContract,
)
from cookiecutter_hypermodern_python_ml_example.contracts.contract_transformer import (
    StandardScalerContract,
)


STANDARD_SCALER_PARAMS = EstimatorContract(
    name="standard_scaler",
    columns=[0],
    estimator_kwargs=StandardScalerContract(),
)

COLUMN_TRANSFORMER_PARAMS = ColumnTransformerContract()


class Transformer:
    """Wrapper for creating a custom Transformer object.

    Based on the scikit-learn `ColumnTransformer
    <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_.
    """

    def __init__(self) -> None:
        """Initialise the Transformer."""
        self.column_transformer: ColumnTransformer = None

    def fit(
        self,
        X: Union[pd.DataFrame, npt.NDArray[Any]],
        y: Optional[Union["pd.Series[int]", npt.NDArray[Any]]] = None,
    ) -> "Transformer":
        """Fit a Transformer using X.

        Args:
            X (Union[pd.DataFrame, npt.NDArray[Any]]): Input data.
            y (Union[pd.Series, npt.NDArray[Any]], optional): Target. Defaults to None.

        Returns:
            Transformer: This Transformer.
        """
        self.column_transformer = ColumnTransformer(
            [
                (
                    STANDARD_SCALER_PARAMS.name,
                    StandardScaler(
                        **STANDARD_SCALER_PARAMS.estimator_kwargs.dict(by_alias=True)
                    ),
                    STANDARD_SCALER_PARAMS.columns,
                )
            ],
            **COLUMN_TRANSFORMER_PARAMS.dict()
        )
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X: Union[pd.DataFrame, npt.NDArray[Any]]) -> Any:
        """Transform X using the Transformer.

        Args:
            X (Union[pd.DataFrame, npt.NDArray[Any]]): Input data with a shape of
            (n_samples, n_features)

        Returns:
            Any: Transformed data with a shape of (n_samples, sum_n_components),
            where sum_n_components is the sum of the output dimensions of each
            component of the ColumnTransformer.
        """
        return self.column_transformer.transform(X)

    def fit_transform(
        self,
        X: Union[pd.DataFrame, npt.NDArray[Any]],
        y: Optional[Union["pd.Series[int]", npt.NDArray[Any]]] = None,
    ) -> Any:
        """Fit a Transformer using X, then transform X using the Transformer.

        Args:
            X (Union[pd.DataFrame, npt.NDArray[Any]]): Input data with a shape of
            (n_samples, n_features)
            y (Union[pd.Series, npt.NDArray], optional): Target. Defaults to None.

        Returns:
            Any: Transformed data with a shape of (n_samples, sum_n_components),
            where sum_n_components is the sum of the output dimensions of each
            component of the ColumnTransformer.
        """
        return self.column_transformer.fit(X, y).transform(X)

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> Any:
        """Get output feature names for transformation.

        Args:
            input_features (Sequence[str], optional): Input features. Defaults to None.

        Returns:
            Any: Transformed feature names.
        """
        return self.column_transformer.get_feature_names_out(input_features)
