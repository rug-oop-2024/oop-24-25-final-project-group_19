from sklearn.linear_model import LinearRegression
from autoop.core.ml.model import Model


class MultipleLinearRegression(Model):
    """
    MultipleLinearRegression is a wrapper of sklearn's Linear Regression.
    """
    def __init__(self) -> None:
        """Initializes a MultipleLinearRegression class"""
        self._model = LinearRegression()
        self._type = "regression"
        super().__init__()
