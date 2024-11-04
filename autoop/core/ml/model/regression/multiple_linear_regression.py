import numpy as np
from autoop.core.ml.model import Model


class MultipleLinearRegression(Model):
    """
    MultipleLinearRegression uses the formula w = (X.T * X)^-1 * X.T * y.
    This formula is used to fit the model and find the the prediction,
    intercept, and coefficients.
    """

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth.

        Parameters:
            observations: An array of training data.
            ground_truth : An array of labels for the training data.
        """
        if observations.ndim == 1:
            observations = observations.reshape(-1, 1)
        elif observations.shape[0] < observations.shape[1]:
            observations = observations.T

        vertex_of_ones = np.ones((observations.shape[0], 1))
        homogeneous_matrix = np.hstack([observations, vertex_of_ones])
        
        transposed_matrix = homogeneous_matrix.T
        covariance_matrix = np.dot(transposed_matrix, homogeneous_matrix)
        inversed_covariance = np.linalg.pinv(covariance_matrix)
        observations_ground_truth = np.dot(transposed_matrix, ground_truth)
        weight = np.dot(inversed_covariance, observations_ground_truth)

        self.parameters = {"weight": weight}

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the output based on the input training data.

        Parameters:
            observations: An array of training data.

        Returns:
            np.ndarray: an array with the predicted values
        """
        if observations.ndim == 1:
            return observations.reshape(-1, 1)

        if observations.shape[0] < observations.shape[1]:
            observations = observations.T

        if observations.shape[1] != self.parameters["weight"].shape[0] - 1:
            raise ValueError("Mismatch in number of features.")

        vertex_of_ones = np.ones((observations.shape[0], 1))
        homogeneous_matrix = np.hstack([observations, vertex_of_ones])

        return np.dot(homogeneous_matrix, self.parameters["weight"])
