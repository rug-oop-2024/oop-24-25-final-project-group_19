class Feature():
    """
    Represents a feature (column) in a dataset, with its name, type, and values
    """
    def __init__(self, name: str, type: str):
        """
        Initializes a Feature object.

        Args:
            name (str): The name of the feature.
            type (str): The type of the feature.
        """
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        """Returns the name of the feature."""
        return self._name

    @property
    def type(self) -> str:
        """Returns the type of the feature."""
        return self._type

    def __str__(self) -> str:
        """Returns a string representation of the Feature."""
        return f"Feature(name = {self._name}, data_type = {self._type})"
