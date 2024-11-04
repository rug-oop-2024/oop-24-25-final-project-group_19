import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types



st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

st.write(datasets)

###  ST/modelling/datasets/list ###

dataset_names = [dataset.name for dataset in datasets]

selected_dataset_name = st.selectbox("Select a Dataset", dataset_names)

selected_dataset = next((dataset for dataset in datasets if dataset._name == selected_dataset_name), None)

###  ST/modelling/datasets/features ###
import io
from autoop.core.ml.artifact import Artifact


if selected_dataset:
    data = pd.read_csv(io.BytesIO(selected_dataset.data)) ####MAYBE USE THIS MAYBE WE CANT###
    st.write(f"Selected Dataset: {selected_dataset_name}")
    st.write(data)
    #We dont want to use this, but this is the only way it works, when we use .read() the 
    #data comes back as encoded and as a artifact, so we cant use detect_feature_types().
    list_of_feature_names = data.columns.tolist()

    selected_input_feature = st.multiselect("Select the input features", list_of_feature_names)
if selected_input_feature:
    features_not_in_input = [feature for feature in list_of_feature_names if feature not in selected_input_feature]
    selected_target_feature = st.selectbox("Select the target feature", features_not_in_input)



