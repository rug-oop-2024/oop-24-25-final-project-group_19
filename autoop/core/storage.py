from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Raises an error when the specific path is not found"""

    def __init__(self, path: str) -> None:
        """Initializes a NotFoundError"""
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract base class for storage """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    Implementation of a Local Storage that manages files.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """Initializes a LocalStorage object."""
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a local file.

        Args:
            data (bytes): The data to be saved.
            key (str): The key is where the data is saved.
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Loads data from a local file.

        Args:
            key (str): The key is where the data is saved.

        Returns:
            bytes: The data read from the file.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete a local file or directory.

        Args:
            key (str): The key is where the data is deleted from.
        """
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """
        List all files under a specified directory path.

        Args:
            prefix (str):  The files are listed under the key.

        Returns:
            List[str]: A list of file paths under the specified prefix.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return list(filter(os.path.isfile, keys))

    def _assert_path_exists(self, path: str) -> None:
        """
        Raise NotFoundError if the specified path does not exist.

        Args:
            path (str): The path to check for existence.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with a relative path.

        Args:
            path (str): The relative path to join.

        Returns:
            str: The combined path of the base path and the relative path.
        """
        return os.path.join(self._base_path, path)
