
from abc import ABC
import numpy as np
from typing import Dict, TypeVar
from copy import deepcopy
from sklearn.preprocessing import StandardScaler


class Model(ABC):
    """
    Abstract base class for the fit and predict methods.
    """
    def __init__(self) -> None:
        """Initializes a Model class with an empty parameters dictionary"""
        self._parameters = {}

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
        Predict the output based on the input observations.

        Args:
            observations (np.ndarray): An array of data for prediction.

        Returns:
            np.ndarray: An array of predicted values.
        """
        return deepcopy(self._model.predict(observations))

    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """
        Retrieves the model's parameters as a dictionary.

        Returns:
            Dict[str, np.ndarray]: A deep copy of the internal dictionary
            containing the model's parameters.
        """
        return deepcopy(self._parameters)

    def add_parameters(
            self,
            key: str,
            value: np.ndarray | list[str] | StandardScaler) -> None:
        """
        Adds or updates a single parameter in the model's parameters.

        Args:
            key (str): The name of the parameter to add or update.
            value (Any): The value of the parameter.
        """
        self._parameters[key] = value

    @property
    def type(self) -> str:
        """Returns the type of the model."""
        return self._type

    ModelObject = TypeVar('Model', bound="Model")

    def _assign_sklearn_parameters(self, model: ModelObject) -> None:
        self.add_parameters("intercept", model.intercept_)
        self.add_parameters("coefficients", model.coef_)
