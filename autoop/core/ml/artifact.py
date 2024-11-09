from copy import deepcopy
from base64 import b64encode
import numpy as np


class Artifact():
    """ Stores and manages metadata about assets. """
    def __init__(
            self,
            name: str,
            asset_path: str,
            version: str,
            data: bytes | np.ndarray,
            type: str,
            metadata: str = None,
            tags: str = None
    ) -> None:
        """
        Initializes an Artifact object.

        Args:
            name (str): Name of the artifact
            asset_path (str): File path where the artifact is stored
            version (str): The version of the artifact.
            data (bytes): The artifact data in bytes format
            type (str): The type of the artifact
            metadata (str): The artifact metadata
            tags (str): The artifact tags
        """
        self._name = name
        self._asset_path = asset_path
        self._version = version
        self._data = data
        self._type = type
        self._metadata = metadata
        self._tags = tags

    @property
    def id(self) -> str:
        """Generates a unique ID based on the asset path and version."""
        encoded_path = b64encode(self._asset_path.encode()).decode()
        return f"{encoded_path}:{self._version}"

    @property
    def data(self) -> bytes | np.ndarray:
        """Provides access to the stored data."""
        return deepcopy(self._data)

    @property
    def asset_path(self) -> str:
        """Provides access to the asset path."""
        return self._asset_path

    @property
    def name(self) -> str:
        """Provides access to the name."""
        return self._name

    @property
    def type(self) -> str:
        """Provides access to the type."""
        return self._type

    @property
    def version(self) -> str:
        """Provides access to the version."""
        return self._version

    @property
    def tags(self) -> str:
        """Provides access to the tags"""
        return self._tags

    @property
    def metadata(self) -> str:
        """Provides access to the metadata."""
        return self._metadata

    def read(self) -> bytes | np.ndarray:
        """Returns the stored data."""
        return deepcopy(self._data)

    def save(self, data: bytes) -> bytes | np.ndarray:
        """Stores the provided data in the artifact and returns it."""
        self._data = data
        return deepcopy(self._data)

    def __repr__(self) -> str:
        """Returns a string representation of the Artifact."""
        return (
            f"Artifact(name={self._name}, "
            f"id={self.id}, "
            f"version={self._version}, "
            f"type={self._type})"
        )
