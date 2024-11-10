from sklearn.linear_model import Lasso as SklearnLasso
from autoop.core.ml.model import Model


class Lasso(Model):
    """
    A Lasso regression model that fits observations to ground truth.
    """
    def __init__(self) -> None:
        """Initializes a Lasso class"""
        self._model = SklearnLasso(alpha=0.01)
        self._type = "regression"
        super().__init__()
