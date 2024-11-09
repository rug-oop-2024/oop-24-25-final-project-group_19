import numpy as np
from sklearn.linear_model import RidgeClassifier
from autoop.core.ml.model import Model
from copy import deepcopy


class RidgeClassificationModel(Model):
    """
    Ridge Classifier model for categorical data.
    """
    def __init__(self) -> None:
        """
        Initializes the Ridge Classifier model.
        """
        self._model = RidgeClassifier()
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
            np.ndarray: an array with the predicted values
        """
        return deepcopy(self._model.predict(observations))
