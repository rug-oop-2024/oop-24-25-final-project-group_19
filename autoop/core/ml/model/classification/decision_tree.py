import numpy as np
from sklearn.tree import DecisionTreeClassifier
from autoop.core.ml.model import Model


class DecisionTreeModel(Model):
    """
    Decision Tree classifier model for categorical data.
    """
    def __init__(self, max_depth: int = 10) -> None:
        """
        Initializes the Decision Tree model, with a default max_depth of 10.
        Additionally to prevent overfitting and long calculation times,
        its not possible to set the max_depth to more than 50.

        Args:
            max_depth (int): The maximum depth of the tree. Default is 10.

        """
        if not isinstance(max_depth, int):
            raise TypeError("max_depth must be an integer.")
        if max_depth <= 0 or max_depth > 50:
            raise ValueError("max_depth must be within the range 1 to 50.")
        self._model = DecisionTreeClassifier(max_depth=max_depth)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth.

        Args:
            observations (np.ndarray): An array of training data
            ground_truth (np.ndarray): An array of labels for the training data
        """
        self._model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the output based on the input training data.

        Args:
            observations (np.ndarray): An array of data for prediction.

        Returns:
            np.ndarray: an array with the predicted values
        """
        return self._model.predict(observations)
