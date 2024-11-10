import streamlit as st
import re
from typing import List, Tuple
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import (
    REGRESSION_MODELS, CLASSIFICATION_MODELS, get_model)
from autoop.core.ml.metric import (
    METRICS_CLASSIFICATION, METRICS_REGRESSION, get_metric)
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.feature import Feature
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
from autoop.core.ml.metric import Metric
import pickle
st.set_page_config(page_title="Modelling", page_icon="📈")


class Modelling:
    """
    A class to manage the creation, execution, and saving of
    pipelines in a Streamlit app.
    """
    def __init__(self) -> None:
        """
        Initialize Modelling instance with default pipeline and summary
        storage.
        """
        self._pipeline = None

    def _write_helper_text(self, text: str) -> None:
        """Write helper text in a styled format.

        Args:
            text (str): The text to display.
        """
        st.write(
            f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

    def _choose_dataset(self) -> None | Dataset:
        """
        Display a dropdown to select a dataset from the registry
        and return the selected dataset.

        Returns:
            Dataset: The selected dataset as a Dataset object.
        """
        automl = AutoMLSystem.get_instance()
        datasets = automl.registry.list(type="dataset")
        datasets_names = [dataset.name for dataset in datasets]
        dataset_dropdown = st.selectbox("Select a Dataset", datasets_names)
        selected_dataset_object = next(
            (dataset for dataset in datasets
                if dataset._name == dataset_dropdown), None)
        if selected_dataset_object:
            selected_dataset = Dataset.initialize_from_line(
                data=selected_dataset_object.data,
                name=selected_dataset_object.name,
                asset_path=selected_dataset_object.asset_path,
                version=selected_dataset_object.version
            )
            st.write(f"Selected Dataset: {selected_dataset.name}")
            return selected_dataset
        return None

    def _choose_features(
            self, dataset: Dataset) -> Tuple[None | List[Feature]]:
        """Allow the user to select input and target features from the dataset.

        Args:
            dataset (Dataset): The dataset to select features from.

        Returns:
            tuple: (input_features, target_feature) as lists or None
            if not selected.
        """
        data = dataset.read()
        if st.button("View the data"):
            st.write(data)
        all_feature_names = data.columns.tolist()
        input_feature_dropdown = st.multiselect(
            "Select the input features", all_feature_names)
        selected_input_feature = self._find_feature(
            input_feature_dropdown, dataset)
        if selected_input_feature:
            features_not_in_input = [
                feature for feature in all_feature_names
                if feature not in input_feature_dropdown]
            target_feature_dropdown = st.selectbox(
                "Select the target feature", features_not_in_input)
            selected_target_features = self._find_feature(
                [target_feature_dropdown], dataset)
            selected_target_feature = (
                selected_target_features[0]
                if selected_target_features else None)
            return selected_input_feature, selected_target_feature
        return None, None

    def _find_feature(
            self, feature_names: list, dataset: Dataset) -> List[Feature]:
        """Find specific features in the dataset based on feature names.

        Args:
            feature_names (list): List of feature names to find.
            dataset (Dataset): The dataset to find features in.

        Returns:
            list: List of matched Feature objects.
        """
        features = detect_feature_types(dataset)
        return [
            feature for feature in features if feature.name in feature_names]

    def _choose_model(self, selected_target_feature: Feature) -> None | Model:
        """
        Allow user to select a model based on the type of target feature.

        Args:
            selected_target_feature (Feature): The selected target feature
            to determine model type.

        Returns:
            Model: The selected model object or None if no model selected.
        """
        if not selected_target_feature:
            return None
        if selected_target_feature.type == "categorical":
            model_dropdown = st.selectbox(
                "What classification model do you want to use?",
                CLASSIFICATION_MODELS)
            selected_model = get_model(model_dropdown)
            self._compatible_metrics = METRICS_CLASSIFICATION
        elif selected_target_feature.type == "numerical":
            model_dropdown = st.selectbox(
                "What regression model do you want to use?", REGRESSION_MODELS)
            selected_model = get_model(model_dropdown)
            self._compatible_metrics = METRICS_REGRESSION
        return selected_model

    def _choose_split_value(self) -> float:
        """
        Prompt the user to select a split value for training and testing data.

        Returns:
            float: The selected split ratio as a decimal value.
        """
        split_value = st.number_input(
            "How would you like to split the dataset?",
            min_value=10,
            max_value=90,
            value=80) / 100
        st.write(
            f"The dataset will be split so that {round(split_value * 100)}% of"
            f" the dataset will be used for training, and "
            f"{round((1 - split_value) * 100)}% of the data will be "
            "used for testing."
        )
        return split_value

    def _choose_metric(self) -> List[Metric]:
        """Allow user to select evaluation metrics for the model.

        Returns:
            list: List of selected metric objects.
        """
        metric_dropdown = st.multiselect(
            "What metric would you like to use?", self._compatible_metrics)
        selected_metric = [get_metric(metric) for metric in metric_dropdown]
        return selected_metric

    def _run_pipeline(
            self,
            selected_model: Model,
            selected_metrics: List[Metric],
            dataset: Dataset,
            input_features: List[Feature],
            target_feature: Feature,
            split_value: float) -> None:
        """Execute the pipeline with selected configurations.

        Args:
            selected_model (Model): The chosen model.
            selected_metrics (list): List of selected metrics.
            dataset (Dataset): The dataset to use in the pipeline.
            input_features (list): List of input features.
            target_feature (Feature): The target feature for prediction.
            split_value (float): The train-test split ratio.
        """
        if st.button("Run model"):
            st.session_state._run_model = True
            st.session_state._pipeline = Pipeline(
                metrics=selected_metrics,
                dataset=dataset,
                model=selected_model,
                input_features=input_features,
                target_feature=target_feature,
                split=split_value
            )
            st.write("### Pipeline Summary")
            self._print_nicely(str(st.session_state._pipeline))

            results = st.session_state._pipeline.execute()
            for key, value in results.items():
                if key == "metrics":
                    st.write("### Metrics")
                    for subset, metrics in value.items():
                        for metric in metrics:
                            metric_name, metric_value = metric
                            st.write(
                                f"**{metric_name.__class__.__name__} "
                                f"for {subset}:** {metric_value}"
                            )
                elif key == "predictions":
                    st.write("### Predictions")
                    for subset, predictions in value.items():
                        st.write(f"**{subset.capitalize()} {key}:**")
                        st.write(predictions)

    def _print_nicely(self, pipeline_str: str) -> None:
        """Format the summary of the pipeline.

        Args:
            pipeline_str (str): The string representation of the pipeline.
        """
        model_type = re.search(
            r"model=(\w+)", pipeline_str).group(1).capitalize()

        input_features_match = re.search(
            r"input_features=\[(.*?)\]", pipeline_str).group(1)
        input_features = re.findall(
            r"Feature\(name = ([\w\s/]+), data_type = (\w+)\)",
            input_features_match)
        input_features_formatted = ', '.join(
            [f"{name.capitalize()} ({data_type})" for name,
             data_type in input_features])

        target_feature_match = re.search(
            r"target_feature=Feature\(name = ([\w\s/]+), data_type = (\w+)\)",
            pipeline_str)
        target_feature = target_feature_match.group(1).capitalize()
        target_feature_type = target_feature_match.group(2).capitalize()

        split_value = re.search(r"split=([\d\.]+)", pipeline_str).group(1)

        metrics_match = re.search(r"metrics=\[(.*?)\]", pipeline_str).group(1)
        metrics = re.findall(
            r"<autoop\.core\.ml\.metric\.(\w+) object", metrics_match)
        metrics_formatted = ', '.join(metrics)

        st.session_state._summary = [
            f"**Type of model:** {model_type}",
            f"**Input features:** {input_features_formatted}",
            f"**Target feature:** {target_feature} ({target_feature_type})",
            f"**Split ratio:** {split_value}",
            f"**Metrics:** {metrics_formatted}"
        ]
        for line in st.session_state._summary:
            st.write(line)

    def _save(self) -> None:
        """Save the current pipeline model as an artifact."""
        if st.session_state.get("_run_model"):
            if "_save" not in st.session_state:
                st.session_state["_save"] = False

            if st.session_state._run_model and (
                st.session_state._save or st.button("SAVE")
            ):
                st.session_state._save = True
                name = st.text_input("Name")
                version = st.text_input("Version", "1.0.0")
                if name and not isinstance(name, str):
                    raise TypeError("Name must be a string")
                if version and not re.match(r"^\d+\.\d+\.\d+$", version):
                    raise ValueError("Version should be in format x.y.z")
                if st.button("Confirm Save"):

                    artifact = Artifact(
                        name=f"{name} ({version})",
                        version=version,
                        data=pickle.dumps(st.session_state._pipeline.model),
                        type="pipeline",
                        asset_path=f"pipeline/{name}",
                        metadata=st.session_state._summary
                    )
                    automl = AutoMLSystem.get_instance()
                    automl.registry.register(artifact)
                    st.write(f"The pipeline '{name}'"
                             f"version: {version} has been saved.")
                    st.session_state._save = False
                    st.session_state._run_model = False

    def run(self) -> None:
        """Main method to execute all modelling steps in the correct order."""
        st.write("# ⚙ Modelling")
        self._write_helper_text(
            "In this section, you can design a machine learning"
            "pipeline to train a model on a dataset.")

        dataset = self._choose_dataset()
        if dataset:
            input_features, target_feature = self._choose_features(dataset)
            if input_features and target_feature:
                selected_model = self._choose_model(target_feature)
                if selected_model:
                    split_value = self._choose_split_value()
                    selected_metrics = self._choose_metric()
                    if selected_metrics:
                        self._run_pipeline(
                            selected_model,
                            selected_metrics,
                            dataset,
                            input_features,
                            target_feature,
                            split_value)
                        self._save()


if __name__ == '__main__':
    modelling_page = Modelling()
    modelling_page.run()
