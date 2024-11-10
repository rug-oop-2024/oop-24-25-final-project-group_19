from sklearn.linear_model import Ridge
from autoop.core.ml.model import Model


class RidgeRegressionModel(Model):
    """A Ridge regression model that fits observations to ground truth."""
    def __init__(self) -> None:
        """
        Initializes a Ridge regression model.
        """
        self._model = Ridge()
        self._type = "regression"
        super().__init__()
