
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification import (
    RidgeClassificationModel,
    LogisticRegressionWrapper,
    PerceptronModel
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
    "ridge_classifier",
    "perceptron_model",
    "logistic_regression"
]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name."""
    model_classes = {
        "ridge_regression": RidgeRegressionModel,
        "lasso": Lasso,
        "elastic_net": ElasticNetWrapper,
        "ridge_classifier": RidgeClassificationModel,
        "perceptron_model": PerceptronModel,
        "logistic_regression": LogisticRegressionWrapper,
        "multiple_linear_regression": MultipleLinearRegression
    }
    if model_name in model_classes:
        return model_classes[model_name]()
