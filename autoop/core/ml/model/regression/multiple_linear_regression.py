import numpy as np
from sklearn.linear_model import LinearRegression
from autoop.core.ml.model import Model
from copy import deepcopy


class MultipleLinearRegression(Model):
    """
    MultipleLinearRegression is a wrapper of sklearn's Linear Regression.
    """
    def __init__(self) -> None:
        self._model = LinearRegression()
        self._type = "regression"
        super().__init__()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth.

        Parameters:
            observations: An array of training data.
            ground_truth : An array of labels for the training data.
        """
        self._model.fit(observations, ground_truth)
        self._assign_sklearn_parameters(self._model)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the output based on the input training data.

        Parameters:
            observations: An array of training data.

        Returns:
            np.ndarray: an array with the predicted values
        """
        return deepcopy(self._model.predict(observations))
