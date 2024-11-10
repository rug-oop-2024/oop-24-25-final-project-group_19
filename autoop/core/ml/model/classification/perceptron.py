from sklearn.linear_model import Perceptron
from autoop.core.ml.model import Model


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
