"""An ML model."""

from typing import Dict, Any
import numpy as np


class Model:
    """Represents an ML model that can be trained and make predictions."""

    def __init__(self, hyperparams: Dict[str, Any]) -> None:
        """Instantiates the object."""
        raise NotImplementedError('Model is abstract.')

    def train(self):
        """Trains the model."""
        raise NotImplementedError('Method is abstract.')

    def predict(self, x_filenames: np.ndarray) -> np.ndarray:
        """Makes a prediction on the given examples."""
        raise NotImplementedError('Method is abstract.')

    def save_to_file(self, filename: str) -> None:
        """Saves the model to the given filename."""
        raise NotImplementedError('Method is abstract.')

    def save_to_dir(self, dir: str) -> None:
        """Saves the model to the given directory. Automatically gives
        the model a unique filename."""
        raise NotImplementedError('Method is abstract.')

    @classmethod
    def load_from_file(cls, filename) -> 'Model':
        """Returns the model instance stored at filename."""
        raise NotImplementedError('Method is abstract.')
