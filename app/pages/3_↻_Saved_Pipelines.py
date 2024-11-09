import streamlit as st
import pandas as pd
import pickle
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Saved Pipelines", page_icon="↻")


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
        dropdown_pipeline = st.selectbox(
            "Choose a saved pipeline", pipeline_names)
        selected_pipeline = pipeline_names[dropdown_pipeline]
        st.session_state._model = pickle.loads(selected_pipeline.data)
        st.write("## Chosen Pipeline summary:")
        st.write(selected_pipeline.metadata)

    def _prediction(self) -> None:
        """
        Allows the user to upload a CSV file and make 
        predictions with the selected pipeline.
        """
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            dataset = Dataset.from_dataframe(
                data = df, 
                name = uploaded_file.name, 
                asset_path = None
                    )
            data = dataset.read()
            if st.button("View the data"):
                st.write(data)
            input_feature_dropdown = st.multiselect(
                "Select the input features", data.columns.tolist()
                    )
            st.write(st.session_state._model.parameters)
            if input_feature_dropdown:
                input_data = data[input_feature_dropdown].to_numpy()
                if ["_prediction"] in st.session_state or st.button("predict"):
                    amount_of_coefficients = len(
                        st.session_state._model.parameters["coefficients"][0]
                            )
                    if len(input_feature_dropdown) != amount_of_coefficients:
                        raise ValueError(
                            f"There should be {amount_of_coefficients}"
                            f"input features."
                        )
                    prediction = st.session_state._model.predict(input_data)
                    st.session_state["_prediction"] = True
                    st.write(prediction)
           
    def run(self) -> None:
        """
        Runs the class by initializing its methods.
        """
        self._get_existing_pipelines()
        self._prediction()

pipelines = saved_pipelines()
pipelines.run()
