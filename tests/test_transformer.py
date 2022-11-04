"""
Test the Transformer.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification

from cookiecutter_hypermodern_python_ml_example.pipeline.transformer import Transformer


clf_X_arr, clf_y_arr = make_classification()
clf_X_df = pd.DataFrame(clf_X_arr, columns=[f"feat_{i}" for i in range(len(clf_X_arr))])
clf_y_df = pd.Series(clf_y_arr)


def test_transformer():
    transformer = Transformer()

    transformer.fit(clf_X_arr, clf_y_arr)
    assert isinstance(transformer.column_transformer, ColumnTransformer)

    transformed_X = transformer.transform(clf_X_arr)
    assert transformed_X.shape[0] == clf_X_arr.shape[0]
    assert transformed_X.shape[1] == len(transformer.get_feature_names_out())

    transformed_X = transformer.fit_transform(clf_X_arr, clf_y_arr)
    assert isinstance(transformer.column_transformer, ColumnTransformer)
    assert transformed_X.shape[0] == clf_X_arr.shape[0]
    assert transformed_X.shape[1] == len(transformer.get_feature_names_out())

    transformer.fit(clf_X_df, clf_y_df)
    assert isinstance(transformer.column_transformer, ColumnTransformer)

    transformed_X = transformer.transform(clf_X_df)
    assert transformed_X.shape[0] == clf_X_df.shape[0]
    assert transformed_X.shape[1] == len(transformer.get_feature_names_out())

    transformed_X = transformer.fit_transform(clf_X_df, clf_y_df)
    assert isinstance(transformer.column_transformer, ColumnTransformer)
    assert transformed_X.shape[0] == clf_X_df.shape[0]
    assert transformed_X.shape[1] == len(transformer.get_feature_names_out())
