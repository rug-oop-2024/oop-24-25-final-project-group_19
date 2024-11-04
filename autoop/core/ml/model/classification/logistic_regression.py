from sklearn.linear_model import LogisticRegression
from autoop.core.ml.model import Model
import numpy as np


class LogisticRegressionWrapper(Model):
    """
    A simplified wrapper for scikit-learn's LogisticRegression model.
    """
    def __init__(self, max_iter: int = 100):
        """
        Initializes the LogisticRegressionWrapper with
        specified maximum iterations.

        Args:
            max_iter (int): Maximum number of iterations for convergence.

        """
        if not isinstance(max_iter, int):
            raise TypeError("max_iter must be an integer.")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        self.model = LogisticRegression(max_iter=max_iter)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth.

        Args:
            observations (np.ndarray): An array of training data
            ground_truth (np.ndarray): An array of labels for the training data
        """
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the output based on the input training data.

        Args:
            observations (np.ndarray): An array of data for prediction.

        Returns:
            np.ndarray: an array with the predicted values
        """
        return self.model.predict(observations)
