import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    dataset = Dataset.from_dataframe(data=df, name=uploaded_file.name, asset_path=f"datasets/{uploaded_file.name}")
    automl.registry.register(dataset)
    st.success("Dataset has been uploaded!")
