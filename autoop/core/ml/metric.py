from abc import ABC, abstractmethod
from typing import TypeVar
import numpy as np

METRICS_CLASSIFICATION = [
    "accuracy",
    "error_rate",
    "jaccard_index"
]

METRICS_REGRESSION = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error"
]
Metric = TypeVar('Metric')


def get_metric(name: str) -> Metric:
    """
    Factory function to get a metric by name and return a metric
    instance given its str name.

    Args:
        name (str): The name of the metric to retrieve.

    Returns:
        Metric: The requested metric class.
    """
    metric_classes = {
        "mean_squared_error": MeanSquaredError,
        "accuracy": Accuracy,
        "error_rate": ErrorRate,
        "mean_absolute_error": MeanAbsoluteError,
        "root_mean_squared_error": RootMeanSquaredError,
        "jaccard_index": JaccardIndex

    }
    if name in metric_classes:
        return metric_classes[name]()


class Metric(ABC):
    """Base class for all metrics."""
    @abstractmethod
    def __call__(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """
        Calculate the metric based on provided ground truth and predictions.

        Args:
            ground_truth (np.ndarray): Array of true labels.
            predictions (np.ndarray): Array of predicted labels.
        Returns:
            float: The calculated metric value.
        """
        pass

    def evaluate(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """
        Wrapper for __call__, allows the use of evaluate() in the Pipeline.
        """
        return self(predictions, ground_truth)


class Accuracy(Metric):
    """Calculates accuracy for classification tasks."""

    def __call__(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """
        Calculate the accuracy of predictions compared to ground truth.

        Args:
            ground_truth (np.ndarray): Array of true labels.
            predictions (np.ndarray): Array of predicted labels.

        Returns:
            float: The accuracy as a proportion of correct predictions.
        """
        return np.mean(ground_truth == predictions)


class ErrorRate(Metric):
    """Calculates error rate for classification tasks."""
    def __call__(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """
        Calculate the error rate of predictions compared to ground truth.

        Args:
            ground_truth (np.ndarray): Array of true labels.
            predictions (np.ndarray): Array of predicted labels.

        Returns:
            float: The error as a proportion of correct predictions.
        """
        return np.mean(ground_truth != predictions)


class JaccardIndex(Metric):
    """Calculates the Jaccard Index for classification tasks."""

    def __call__(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """
        Calculate the Jaccard Index between ground truth and predictions.

        Args:
            ground_truth (np.ndarray): Array of true labels.
            predictions (np.ndarray): Array of predicted labels.

        Returns:
            float: The Jaccard Index.
        """
        intersection = np.sum(ground_truth == predictions)
        union = len(ground_truth) + len(predictions) - intersection
        return intersection / union if union > 0 else 0.0


class MeanSquaredError(Metric):
    """Calculates Mean Squared Error for regression tasks."""

    def __call__(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """
        Calculate the Mean Squared Error between predictions and ground truth.

        Args:
            ground_truth (np.ndarray): Array of true values.
            predictions (np.ndarray): Array of predicted values.

        Returns:
            float: The Mean Squared Error.
        """
        return np.mean((ground_truth - predictions) ** 2)


class RootMeanSquaredError(Metric):
    """Calculates Root Mean Squared Error for regression tasks."""

    def __call__(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """
        Calculate the Root Mean Squared Error between predictions
        and ground truth.

        Args:
            ground_truth (np.ndarray): Array of true values.
            predictions (np.ndarray): Array of predicted values.

        Returns:
            float: The Root Mean Squared Error.
        """
        return np.sqrt(np.mean((ground_truth - predictions) ** 2))


class MeanAbsoluteError(Metric):
    """Calculates Mean Absolute Error for regression tasks. """

    def __call__(
        self,
        ground_truth: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """
        Calculate the Mean Absolute Error between predictions and ground truth.

        Args:
            ground_truth (np.ndarray): Array of true values.
            predictions (np.ndarray): Array of predicted values.

        Returns:
            float: The Mean Absolute Error.
        """
        return np.mean(np.abs(ground_truth - predictions))
