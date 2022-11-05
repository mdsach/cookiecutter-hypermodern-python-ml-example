"""Test the Transformer."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification

from cookiecutter_hypermodern_python_ml_example.pipeline.transformer import Transformer


CLF_ARR_X, CLF_ARR_Y = make_classification()
CLF_DF_X = pd.DataFrame(
    CLF_ARR_X, columns=[f"feat_{i}" for i in range(CLF_ARR_X.shape[1])]
)
CLF_DF_Y = pd.Series(CLF_ARR_Y)


def test_transformer() -> None:
    """Tests fit, transform and fit_transform methods of the Transformer."""
    transformer = Transformer()

    transformer.fit(CLF_ARR_X, CLF_ARR_Y)
    assert isinstance(transformer.column_transformer, ColumnTransformer)

    transformed_x = transformer.transform(CLF_ARR_X)
    assert transformed_x.shape[0] == CLF_ARR_X.shape[0]
    assert transformed_x.shape[1] == len(transformer.get_feature_names_out())

    transformed_x = transformer.fit_transform(CLF_ARR_X, CLF_ARR_Y)
    assert isinstance(transformer.column_transformer, ColumnTransformer)
    assert transformed_x.shape[0] == CLF_ARR_X.shape[0]
    assert transformed_x.shape[1] == len(transformer.get_feature_names_out())

    transformer.fit(CLF_DF_X, CLF_DF_Y)
    assert isinstance(transformer.column_transformer, ColumnTransformer)

    transformed_x = transformer.transform(CLF_DF_X)
    assert transformed_x.shape[0] == CLF_DF_X.shape[0]
    assert transformed_x.shape[1] == len(transformer.get_feature_names_out())

    transformed_x = transformer.fit_transform(CLF_DF_X, CLF_DF_Y)
    assert isinstance(transformer.column_transformer, ColumnTransformer)
    assert transformed_x.shape[0] == CLF_DF_X.shape[0]
    assert transformed_x.shape[1] == len(transformer.get_feature_names_out())
