"""DatasetSequence class."""

import math
from typing import Tuple, Optional, Callable
import numpy as np
from tensorflow.keras.utils import Sequence

from util.util import get_max_batch_size

OVERFIT_ONE_BATCH = False
DEFAULT_BATCH_SIZE = 32
SHUFFLE_ON_EPOCH_END = True
CHECK_MAX_BATCH_SIZE = True


class DatasetSequence(Sequence):
    """Represents a single sequence in the dataset (i.e., the train,
    val, or test set)."""

    def __init__(self, x_filenames: np.ndarray, y: Optional[np.ndarray] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 batch_augment_fn: Optional[Callable] = None,
                 batch_format_fn: Optional[Callable] = None) -> None:
        """Instantiates the object.
        :param x_filenames: ndarray of strs containing the filenames of
        all examples in the dataset.
        :param y: ndarray of ints containing the labels of all examples
        in the dataset. If None, "labels" are set to all zeros as a
        placeholder (use this for predictions, where labels are
        unknown).
        :param batch_size: the number of examples in each batch.
        :param batch_augment_fn: the function to augment a batch of
        data.
        :param batch_format_fn: the function to format a batch of data.
        """
        # pylint: disable=invalid-name
        if y is None:
            y = np.zeros(x_filenames.shape[0])
        if x_filenames.shape[0] != y.shape[0]:
            raise ValueError('Found {0} examples, but {1} labels'.format(
                x_filenames.shape[0], y.shape[0]))
        self.x_filenames: np.ndarray = x_filenames
        self.y: np.ndarray = y
        self.batch_size: int = batch_size
        self.batch_augment_fn: Optional[Callable] = batch_augment_fn
        self.batch_format_fn: Optional[Callable] = batch_format_fn
        if CHECK_MAX_BATCH_SIZE:
            print('Estimated maximum batch size: {0}'.format(
                get_max_batch_size(x_filenames)))

    def __len__(self) -> int:
        """Returns the number of batches in the dataset.
        :return: the length of the dataset.
        """
        return math.ceil(len(self.x_filenames) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a single batch of loaded data.
        :param idx: the batch index.
        :return: a tuple of two np.ndarray objects, the batch x values
        (features) and the batch labels, respectively.
        """
        if OVERFIT_ONE_BATCH:
            idx = 0
        # TODO need to look at the data

    def on_epoch_end(self) -> None:
        """Performs actions that happen at the end of every epoch, e.g.
        shuffling."""
        if SHUFFLE_ON_EPOCH_END:
            # TODO this seems superfluous since model.fit has a shuffle flag. Docs are unclear, so test out.
            self.shuffle()

    def shuffle(self) -> None:
        """Shuffles the dataset."""
        shuffled_indices = np.random.permutation(self.x_filenames.shape[0])
        self.x_filenames, self.y = self.x_filenames[shuffled_indices], \
                                   self.y[shuffled_indices]
