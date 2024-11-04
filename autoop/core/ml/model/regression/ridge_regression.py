import numpy as np
from sklearn.linear_model import Ridge
from autoop.core.ml.model import Model


class RidgeRegressionModel(Model):
    """A Ridge regression model that fits observations to ground truth."""
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initializes a Ridge regression model with the specified regularization
        strength.

        Args:
            alpha (float): Regularization strength. Must be a non-negative
                        float. Larger values specify stronger regularization.
        """
        if not isinstance(alpha, float):
            raise TypeError("Alpha must be a float")
        if alpha <= 0:
            raise ValueError("Alpha can only take positive values")
        self._model = Ridge(alpha=alpha)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth using
        the Ridge regression model.

        Args:
            observations (np.ndarray): An array of training data
            ground_truth (np.ndarray): An array of labels for the training data
        """
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the output based on the input observations using the Ridge
        regression model.

        Args:
            observations (np.ndarray): An array of data for prediction.

        Returns:
            np.ndarray: An array of predicted values.
        """
        return self._model.predict(observations)
