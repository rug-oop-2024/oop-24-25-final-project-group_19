
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification import (
    DecisionTreeModel,
    LogisticRegressionWrapper,
    KNearestNeighbors
)
from autoop.core.ml.model.regression import (
    Lasso,
    ElasticNetWrapper,
    MultipleLinearRegression,
    RidgeRegressionModel
)

REGRESSION_MODELS = [
    "ridge_regression",
    "lasso",
    "elastic_net",
    "multiple_linear_regression"
]

CLASSIFICATION_MODELS = [
    "decision_tree",
    "k_nearest_neighbors",
    "logistic_regression"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    model_classes = {
        "ridge_regression": RidgeRegressionModel,
        "lasso": Lasso,
        "elastic_net": ElasticNetWrapper,
        "decision_tree": DecisionTreeModel,
        "k_nearest_neighbors": KNearestNeighbors,
        "logistic_regression": LogisticRegressionWrapper,
        "multiple_linear_regression": MultipleLinearRegression
    }
    if model_name in model_classes:
        return model_classes[model_name]()
