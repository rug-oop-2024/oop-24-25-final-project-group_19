from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    data = dataset.read()
    list_of_features = []
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            list_of_features.append(Feature(name=column, type="numerical"))
        else:
            list_of_features.append(Feature(name=column, type="categorical"))
    return list_of_features
