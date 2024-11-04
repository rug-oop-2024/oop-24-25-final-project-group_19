from pydantic import BaseModel, Field
import pandas as pd
import base64
from copy import deepcopy

class Artifact():
    """ Stores and manages metadata about assets. """
    def __init__(self, name: str, asset_path: str, version: str, data: bytes, type: str, metadata: str=None, tags: str=None) -> None:
        """
        Initializes an Artifact object.

        Args:
            name (str): Name of the artifact
            asset_path (str): File path where the artifact is stored
            version (str): The version of the artifact.
            data (bytes): The artifact data in bytes format
            type (str): The type of the artifact
        """
        self._name = name
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._type = type
        self._metadata = metadata
        self._tags = tags

    @property
    def id(self):
        """Generates a unique ID based on the asset path and version."""
        encoded_path = base64.urlsafe_b64encode(self._asset_path.encode()).decode()
        return f"{encoded_path}:{self._version}"
    @property
    def data(self) -> bytes:
        """Provides access to the stored data."""
        return self._data
    @property
    def asset_path(self)-> str:
        return self._asset_path

    @property
    def name(self)-> str:
        return self._name
    
    @property
    def type(self) -> str:
        return self._type
    @property
    def version(self)-> str:
        return self._version
    @property
    def tags(self)-> str:
        return self._tags
    @property
    def metadata(self)->str:
        return self._metadata
    
    def read(self) -> bytes:
        """Returns the stored data."""
        return self._data
    
    def save(self, data: bytes) -> bytes:
        """Stores the provided data in the artifact and returns it."""
        self._data = data
        return self._data
    
    def __repr__(self):
        """Returns a string representation of the Artifact."""
        return f"Artifact(name={self._name}, id={self.id}, version={self._version}, type={self._type})"
        