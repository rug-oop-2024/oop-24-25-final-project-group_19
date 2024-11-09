from sklearn.linear_model import ElasticNet
import numpy as np
from autoop.core.ml.model import Model
from copy import deepcopy


class ElasticNetWrapper(Model):
    """
    ElasticNet regression model.
    """
    def __init__(self) -> None:
        """
        Initializes the ElasticNetWrapper.
        """
        self._model = ElasticNet()
        self._type = "regression"
        super().__init__()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fits the ElasticNet model to the training data.

        Args:
            observations (np.ndarray): An array of training data
            ground_truth (np.ndarray): An array of labels for the training data
        """
        self._model.fit(observations, ground_truth)
        self._assign_sklearn_parameters(self._model)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the fitted ElasticNet model.

        Args:
            observations (np.ndarray): An array of data for prediction.

        Returns:
            np.ndarray: An array of predicted values.
        """
        return deepcopy(self._model.predict(observations))
