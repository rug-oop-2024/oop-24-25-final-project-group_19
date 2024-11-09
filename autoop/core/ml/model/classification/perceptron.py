import numpy as np
from sklearn.linear_model import Perceptron
from autoop.core.ml.model import Model
from copy import deepcopy


class PerceptronModel(Model):
    """
    Perceptron classifier model for binary classification.
    """
    def __init__(self) -> None:
        """
        Initializes the Perceptron model.
        """
        self._model = Perceptron()
        self._type = "classification"
        super().__init__()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth.

        Args:
            observations (np.ndarray): An array of training data
            ground_truth (np.ndarray): An array of labels for the training data
        """
        self._model.fit(observations, ground_truth)
        self._assign_sklearn_parameters(self._model)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the output based on the input training data.

        Args:
            observations (np.ndarray): An array of data for prediction.

        Returns:
            np.ndarray: an array with the predicted values.
        """
        return deepcopy(self._model.predict(observations))
