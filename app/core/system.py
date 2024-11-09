from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry():
    """
    Manages registration, retrieval, and deletion of artifacts in the system.
    """
    def __init__(self,
                 database: Database,
                 storage: Storage) -> None:
        """Initialize with database and storage."""
        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """Register and save an artifact in storage and database."""
        self._storage.save(artifact.data, artifact.asset_path)
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set(f"artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """List artifacts, and can filter by type."""
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """Retrieve an artifact by ID from storage."""
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """Delete an artifact by ID from both storage and database."""
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)

    def get_id(self, name: str, version: str) -> str:
        """Retrieve the ID of an artifact based on its name and version."""
        entries = self._database.list("artifacts")
        artifact_id = next(
            (
                artifact_id 
                for artifact_id, data in entries 
                if data["name"] == name and data["version"] == version
            ), None)
        if artifact_id is not None:
            return artifact_id
        raise ValueError(
            f"Artifact with name '{name}' and version '{version}' not found.")


class AutoMLSystem:
    """
    Singleton system that provides a registry and storage
    access for managing artifacts.
    """
    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initialize AutoMLSystem with storage and database,
        and create an artifact registry.
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> None:
        """
        Retrieve the singleton instance of AutoMLSystem or
        create it if necessary.
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"), 
                Database(
                    LocalStorage("./assets/dbo")
                )
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> None:
        """Access the artifact registry."""
        return self._registry
