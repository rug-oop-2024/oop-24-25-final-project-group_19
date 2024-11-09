import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


class uploading_datasets:
    """
    Initialize the DatasetUploader with access to the AutoMLSystem instance.
    """
    def _upload(self) -> None:
        """Handle file upload and dataset registration."""
        automl = AutoMLSystem.get_instance()
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            dataset = Dataset.from_dataframe(
                data=df,
                name=uploaded_file.name,
                asset_path=f"datasets/{uploaded_file.name}"
                )
            automl.registry.register(dataset)
            st.success("Dataset has been uploaded!")

    def run(self) -> None:
        """Execute the upload process."""
        self._upload()


upload = uploading_datasets()
upload.run()
