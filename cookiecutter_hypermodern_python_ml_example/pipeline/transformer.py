from typing import Sequence
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

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


class Transformer(TransformerMixin):
    """Wrapper for creating a custom Transformer object, based on the scikit-learn
    `ColumnTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_.
    """

    def __init__(self):
        self.column_transformer: ColumnTransformer = None

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray] = None
    ) -> "Transformer":
        check_array(X)
        self.transformer = ColumnTransformer(
            [
                (
                    STANDARD_SCALER_PARAMS.name,
                    StandardScaler(**STANDARD_SCALER_PARAMS.estimator_kwargs.dict()),
                    STANDARD_SCALER_PARAMS.columns,
                )
            ],
            **COLUMN_TRANSFORMER_PARAMS.dict()
        )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.column_transformer.transform(X)

    def get_feature_names_out(self, input_features: Sequence[str] = None) -> np.ndarray:
        return self.column_transformer.get_feature_names_out(input_features)
