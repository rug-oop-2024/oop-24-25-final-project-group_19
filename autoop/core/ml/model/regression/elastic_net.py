from sklearn.linear_model import ElasticNet
import numpy as np
from autoop.core.ml.model import Model


class ElasticNetWrapper(Model):
    """
    A wrapper for scikit-learn's ElasticNet regression model with
    input validation.
    """
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5) -> None:
        """
        Initializes the ElasticNetWrapper with specified parameters.

        Args:
            alpha (float): Regularization strength
            l1_ratio (float): Mixing parameter for ElasticNet

        """
        if not isinstance(alpha, float) or not isinstance(l1_ratio, float):
            raise TypeError("alpha and l1_ratio must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if not (0 <= l1_ratio <= 1):
            raise ValueError("l1_ratio must be between 0 and 1.")
        self._model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the ElasticNet model to the training data.

        Args:
            observations (np.ndarray): An array of training data
            ground_truth (np.ndarray): An array of labels for the training data
        """
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the fitted ElasticNet model.

        Args:
            observations (np.ndarray): An array of data for prediction.

        Returns:
            np.ndarray: An array of predicted values.
        """
        return self._model.predict(observations)
