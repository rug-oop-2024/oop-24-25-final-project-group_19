import streamlit as st
import pandas as pd
import numpy as np
import pickle
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types
from autoop.functional.preprocessing import preprocess_features

st.set_page_config(page_title="Deployment", page_icon="↻")


class saved_pipelines:
    """
    A Streamlit app class for loading saved pipelines,
    and making predictions with the chosen pipeline model.
    """
    def _get_existing_pipelines(self) -> None:
        """
        Allows the user to load and display a pipeline from
        a list of saved pipelines.
        """
        automl = AutoMLSystem.get_instance()
        pipelines = automl.registry.list(type="pipeline")
        pipeline_names = {pipeline.name: pipeline for pipeline in pipelines}
        if len(pipeline_names) == 0:
            st.write("There are no saved pipelines yet!")
        else:
            dropdown_pipeline = st.selectbox(
                "Choose a saved pipeline", pipeline_names)
            selected_pipeline = pipeline_names[dropdown_pipeline]
            st.session_state._model = pickle.loads(selected_pipeline.data)
            st.write("## Chosen Pipeline summary:")
            for line in selected_pipeline.metadata:
                st.write(line)

    def _prediction(self) -> None:
        """
        Allows the user to upload a CSV file and make
        predictions with the selected pipeline.
        """
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            dataset = Dataset.from_dataframe(
                data=df,
                name=uploaded_file.name,
                asset_path=None
            )
            data = dataset.read()
            if st.button("View the data"):
                st.write(data)
            features = detect_feature_types(dataset)
            available_features = [feature.name for feature in features]
            input_feature_dropdown = st.multiselect(
                "Select the input features", available_features)
            selected_features = [
                feature for feature in features
                if feature.name in input_feature_dropdown]
            if input_feature_dropdown:
                if ["_prediction"] in st.session_state or st.button("predict"):
                    preprocessed_features = preprocess_features(
                        selected_features, dataset)
                    input_data = np.hstack([data for name, data,
                                            artifact in preprocessed_features])
                    expected_features = len(
                        st.session_state._model.parameters["coefficients"][0])
                    if input_data.shape[1] != expected_features:
                        raise ValueError(
                            f"There should be {expected_features}"
                            f" input features."
                        )
                    prediction = st.session_state._model.predict(input_data)
                    st.session_state["_prediction"] = True
                    if "encoder_list" in st.session_state._model.parameters:
                        encoder_list = st.session_state._model.parameters[
                            "encoder_list"]
                        prediction = [
                            encoder_list[category] for category in prediction]

                    if "scaler" in st.session_state._model.parameters:
                        scaler = st.session_state._model.parameters["scaler"]
                        prediction = scaler.inverse_transform(
                            prediction.reshape(-1, 1)).flatten()
                    st.write(prediction)

    def run(self) -> None:
        """
        Runs the class by initializing its methods.
        """
        self._get_existing_pipelines()
        self._prediction()


if __name__ == '__main__':
    pipelines = saved_pipelines()
    pipelines.run()
