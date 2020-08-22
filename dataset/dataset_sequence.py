"""Contains the DatasetSequence class."""

import math
from typing import List, Tuple
import numpy as np
from tensorflow.keras.utils import Sequence

OVERFIT_ONE_BATCH = False
DEFAULT_BATCH_SIZE = 32
SHUFFLE_ON_EPOCH_END = True


class DatasetSequence(Sequence):
    """Represents a single sequence in the dataset (i.e., the train,
    val, or test set)."""

    def __init__(self, x: List[str], y: List[int],
                 batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """Instantiates the object."""
        # pylint: disable=invalid-name
        self.x: List[str] = x
        self.y: List[int] = y
        self.batch_size: int = batch_size
        pass

    def __len__(self) -> int:
        """Returns the number of batches in the dataset."""
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a single batch of loaded data.
        :param idx: the batch index.
        :return: a tuple of two np.ndarray objects, the batch x values
        (features) and the batch labels, respectively.
        """
        if OVERFIT_ONE_BATCH:
            idx = 0
        # TODO

    def on_epoch_end(self) -> None:
        """Performs actions that happen at the end of every epoch, e.g.
        shuffling."""
        if SHUFFLE_ON_EPOCH_END:
            self.shuffle()

    def shuffle(self) -> None:
        """Shuffles the dataset."""
        # TODO

    def get_max_batch_size(self) -> int:
        """Returns the maximum batch size attainable on the current
        CPU/GPU."""
        # TODO
