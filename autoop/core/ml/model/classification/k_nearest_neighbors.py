import numpy as np
from collections import Counter
from autoop.core.ml.model import Model


class KNearestNeighbors(Model):
    """
    KNearestNeighbors class predicts the label of an unknown data based on the
    'k' nearest observations.
    """
    def __init__(self, k: int = 3) -> None:
        """
        Initializes the KNearestNeighbors model with a specified number
        of neighbors.

        Args:
            k (int): The number of nearest neighbors to consider when
            making predictions. Must be a positive integer (default is 3).
        """
        if not isinstance(k, int):
            raise TypeError("k must be an integer.")
        if k <= 0:
            raise ValueError("k can only take non-zero, positive values ")
        self._k = k
        super().__init__()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth.

        Args:
            observations (np.ndarray): An array of training data
            ground_truth (np.ndarray): An array of labels for the training data

        Returns: None
        """

        self.parameters = {
            "observations": observations,
            "ground_truth": ground_truth
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the output based on the input training data.

        Args:
            observations (np.ndarray): An array of data for prediction.

        Returns:
            np.ndarray: an array with the predicted values
        """
        predictions = [self._predict_single(x) for x in observations]
        return np.array(predictions)

    def _predict_single(self, observation: np.ndarray) -> np.ndarray:
        """
        This function predicts the label of one single observation by
        calculating distances between every observation to find the most
        common labels among 'k'nearest neighbors.

        Arguments:
            observation(np.ndarray): Gives a single data point where its label
            needs to be predicted.

        Returns:
            np.ndarray: This function returns a numpy array with the predicted
            label for the input data.

        """
        distances = np.linalg.norm(
            self.parameters["observations"] - observation, axis=1
        )
        k_indices = np.argsort(distances)[: self._k]
        kn_labels = [self.parameters["ground_truth"][i] for i in k_indices]
        most_common = Counter(
            kn_labels
        ).most_common()
        return most_common[0][0]
