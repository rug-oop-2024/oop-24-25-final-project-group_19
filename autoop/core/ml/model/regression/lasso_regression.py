import numpy as np
from sklearn.linear_model import Lasso as SklearnLasso
from autoop.core.ml.model import Model
from copy import deepcopy


class Lasso(Model):
    """
    A Lasso regression model that fits observations to ground truth.
    """
    def __init__(self) -> None:
        self._model = SklearnLasso()
        self._type = "regression"
        super().__init__()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth, using
        the Lasso model.

        Args:
            observations (np.ndarray): An array of training data
            ground_truth (np.ndarray): An array of labels for the training data
        """
        self._model.fit(observations, ground_truth)
        self._assign_sklearn_parameters(self._model)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the output based on the input training data using the
        Lasso model.

        Args:
            observations (np.ndarray): An array of data for prediction.

        Returns:
            np.ndarray: An array with the predicted values.
        """
        return deepcopy(self._model.predict(observations))
