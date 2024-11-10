from autoop.core.ml.artifact import Artifact
import pandas as pd
import io
from typing import TypeVar


class Dataset(Artifact):
    """
    A class representing a dataset artifact which inherits from Artifact class.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a Datase class.
        """
        super().__init__(type="dataset", *args, **kwargs)

    DatasetObject = TypeVar('DatasetType', bound="Dataset")

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame,
        name: str, asset_path: str,
        version: str = "1.0.0"
            ) -> DatasetObject:
        """
        Create a Dataset instance from a pandas DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame to be saved as a dataset.
            name (str): The name of the dataset.
            asset_path (str): The file path where the dataset will be stored.
            version (str): Version of the dataset. Defaults to "1.0.0".

        Returns:
            Dataset: A new Dataset instance with the given data,
            name, and version.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the CSV data

        Returns:
            pd.DataFrame: The dataset in DataFrame format.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Save the dataset as byte-encoded CSV data.

        Args:
            data (pd.DataFrame): DataFrame to be saved.

        Returns:
            bytes: byte-encoded CSV data
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)

    @staticmethod
    def initialize_from_line(
        data: bytes,
        name: str,
        asset_path: str,
        version: str = "1.0.0"
    ) -> DatasetObject:
        """
        Creates a Dataset object from pre-encoded
        bytes data without invoking __init__.

        Args:
            data (bytes): Byte-encoded CSV data.
            name (str): Name of the dataset.
            asset_path (str): Path to store the dataset.
            version (str): Version of the dataset.

        Returns:
            Dataset: A Dataset object created from the byte-encoded data.
        """

        instance = Dataset.__new__(Dataset)

        instance._name = name
        instance._asset_path = asset_path
        instance._data = data
        instance._version = version
        instance._type = "dataset"

        return instance
