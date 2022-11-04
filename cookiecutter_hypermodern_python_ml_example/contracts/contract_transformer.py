"""Contracts for Transformer parameters."""

from typing import Union

from pydantic import BaseModel
from pydantic import StrictBool
from pydantic import StrictFloat
from pydantic import StrictInt
from pydantic import StrictStr


class StandardScalerContract(BaseModel):
    """Class to store parameters for a StandardScaler.

    scikit-learn `StandardScaler
    <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.
    """

    copy: StrictBool = True
    with_mean: StrictBool = True
    with_std: StrictBool = True


class MinMaxScalerContract(BaseModel):
    """Class to store parameters for a MinMaxScaler.

    scikit-learn `MinMaxScaler
    <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_.
    """

    feature_range: tuple[float, float] = (0, 1)
    copy: StrictBool = True
    clip: StrictBool = False


class EstimatorContract(BaseModel):
    """Class to store parameters for each component of a ColumnTransformer.

    scikit-learn `ColumnTransformer
    <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_.
    """

    name: StrictStr
    columns: Union[list[StrictStr], list[StrictInt]]
    estimator_kwargs: Union[StandardScalerContract, MinMaxScalerContract]


class ColumnTransformerContract(BaseModel):
    """Class to store parameters for a ColumnTransformer.

    scikit-learn `ColumnTransformer
    <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_.
    """

    remainder: StrictStr = "drop"
    sparse_threshold: StrictFloat = 0.3
    n_jobs: StrictInt = -1
    transformer_weights: dict[StrictStr, float] = None
    verbose: StrictBool = False
    verbose_feature_names_out: StrictBool = True
