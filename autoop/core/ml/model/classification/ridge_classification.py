from sklearn.linear_model import RidgeClassifier
from autoop.core.ml.model import Model


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
