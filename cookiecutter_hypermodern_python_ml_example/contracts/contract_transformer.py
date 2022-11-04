from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

from pydantic import BaseModel
from pydantic import StrictBool
from pydantic import StrictFloat
from pydantic import StrictInt
from pydantic import StrictStr


class StandardScalerContract(BaseModel):
    """Class to store parameters for a
    `StandardScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_."""

    copy: StrictBool = True
    with_mean: StrictBool = True
    with_std: StrictBool = True


class MinMaxScalerContract(BaseModel):
    """Class to store parameters for a
    `MinMaxScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_.
    """

    feature_range: Tuple[float, float] = (0, 1)
    copy: StrictBool = True
    clip: StrictBool = False


class EstimatorContract(BaseModel):
    """Class to store parameters for a single estimator for use in a
    `ColumnTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_.
    """

    name: StrictStr
    columns: Union[List[StrictStr], List[StrictInt]]
    estimator_kwargs: Union[StandardScalerContract, MinMaxScalerContract]


class ColumnTransformerContract(BaseModel):
    """Class to store parameters for the
    `ColumnTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html>`_.
    """

    remainder: StrictStr = "drop"
    sparse_threshold: StrictFloat = 0.3
    n_jobs: StrictInt = -1
    transformer_weights: Dict[StrictStr, float] = None
    verbose: StrictBool = False
    verbose_feature_names_out: StrictBool = True
