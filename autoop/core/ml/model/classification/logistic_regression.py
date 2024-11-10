from sklearn.linear_model import LogisticRegression
from autoop.core.ml.model import Model


class LogisticRegressionWrapper(Model):
    """
    A simplified wrapper for scikit-learn's LogisticRegression model.
    """
    def __init__(self):
        """
        Initializes the LogisticRegressionWrapper.
        """
        self._model = LogisticRegression()
        self._type = "classification"
        super().__init__()
