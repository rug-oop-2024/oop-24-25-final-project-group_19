import numpy as np
from sklearn.linear_model import Ridge
from autoop.core.ml.model import Model
from copy import deepcopy


class RidgeRegressionModel(Model):
    """A Ridge regression model that fits observations to ground truth."""
    def __init__(self) -> None:
        """
        Initializes a Ridge regression model.
        """
        self._model = Ridge()
        self._type = "regression"
        super().__init__()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth using
        the Ridge regression model.

        Args:
            observations (np.ndarray): An array of training data
            ground_truth (np.ndarray): An array of labels for the training data
        """
        self._model.fit(observations, ground_truth)
        self._assign_sklearn_parameters(self._model)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the output based on the input observations using the Ridge
        regression model.

        Args:
            observations (np.ndarray): An array of data for prediction.

        Returns:
            np.ndarray: An array of predicted values.
        """
        return deepcopy(self._model.predict(observations))
