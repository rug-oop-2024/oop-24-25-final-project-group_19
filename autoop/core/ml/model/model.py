
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict
from copy import deepcopy


class Model(ABC):
    """
    Abstract base class for the fit and predict methods.
    """
    def __init__(self) -> None:
        self._parameters = {}

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Fit the model to the provided observations and ground truth.

        Parameters:
            observations: An array of training data.
            ground_truth : An array of labels for the training data.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the output based on the input training data.

        Parameters:
            observations: An array of training data.

        Returns:
            np.ndarray: an array with the predicted values
        """
        pass

    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """
        Retrieves the model's parameters as a dictionary.

        Returns:
            Dict[str, np.ndarray]: A deep copy of the internal dictionary
            containing the model's parameters.
        """
        return deepcopy(self._parameters)

    def add_parameters(self, key: str, value) -> None:
        """
        Adds or updates a single parameter in the model's parameters.

        Args:
            key (str): The name of the parameter to add or update.
            value (Any): The value of the parameter.
        """
        self._parameters[key] = value

    @property
    def type(self):
        """Returns the type of the model."""
        return self._type

    def _assign_sklearn_parameters(self, model):
        self.add_parameters("intercept", model.intercept_)
        self.add_parameters("coefficients", model.coef_)
