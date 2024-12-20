from typing import List, Dict
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np
from copy import deepcopy


class Pipeline():
    """
    Represents a machine learning pipeline for model training and evaluation.
    """
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """Initializes the pipeline."""
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        model_type = model.type != "classification"
        if (target_feature.type == "categorical" and model_type):
            raise ValueError(
                "Model type must be classification for categorical"
                " target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """Returns a string summary of the pipeline configuration."""
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """Returns the model used in the pipeline."""
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """Returns a list of artifacts generated when pipeline runs."""
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """Registers an artifact to the pipeline."""
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """Preprocesses the input and target features."""
        (target_feature_name, target_data, artifact) = \
            preprocess_features([self._target_feature], self._dataset)[0]

        if target_data.ndim > 1 and target_data.shape[1] > 1:
            target_data = np.argmax(target_data, axis=1).ravel()

        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sort by feature
        # name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact)
                               in input_results]

    def _split_data(self) -> None:
        """Splits data into training and testing sets."""
        split = self._split
        self._train_X = [vector[:int(split * len(vector))] for vector in
                         self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):] for vector in
                        self._input_vectors]
        self._train_y = self._output_vector[:int(split * len(
            self._output_vector))]
        self._test_y = self._output_vector[int(split * len(
            self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Concatenates a list of vectors into a single array."""
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """Trains the model using the training data."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """Evaluates the model on the testing data."""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> Dict[str, Dict[str, List]]:
        """
        Executes the pipeline: preprocesses, splits, trains, and evaluates.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._metrics_results_train = []
        predictions = self._model.predict(X)

        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results_train.append((metric, result))
        self._evaluate()

        if "encoder" in self._artifacts[self._target_feature.name]:
            encoder = self._artifacts[self._target_feature.name]["encoder"]
            category_list = encoder.categories_[0]
            predictions = [category_list[category] for category in predictions]
            self._predictions = [
                category_list[category] for category in self._predictions]
            self._model.add_parameters("encoder_list", category_list)

        if "scaler" in self._artifacts[self._target_feature.name]:
            scaler = self._artifacts[self._target_feature.name]["scaler"]
            predictions = scaler.inverse_transform(
                predictions.reshape(-1, 1)).flatten()
            self._predictions = scaler.inverse_transform(
                self._predictions.reshape(-1, 1)).flatten()
            self._model.add_parameters("scaler", scaler)

        return deepcopy({
            "metrics": {
                "training": self._metrics_results_train,
                "evaluation": self._metrics_results
            },
            "predictions": {
                "training": predictions,
                "evaluation": self._predictions
            }
        })
