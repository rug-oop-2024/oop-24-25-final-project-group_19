from sklearn.linear_model import ElasticNet
from autoop.core.ml.model import Model


class ElasticNetWrapper(Model):
    """
    ElasticNet regression model.
    """
    def __init__(self) -> None:
        """
        Initializes the ElasticNetWrapper.
        """
        self._model = ElasticNet()
        self._type = "regression"
        super().__init__()
