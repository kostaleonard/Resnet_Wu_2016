"""DatasetSequence class."""

import math
from typing import Tuple, Optional, Callable
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from util.util import get_max_batch_size, normalize_images

OVERFIT_ONE_BATCH = False
DEFAULT_BATCH_SIZE = 32
SHUFFLE_ON_EPOCH_END = True
CHECK_MAX_BATCH_SIZE = True
DEFAULT_TARGET_SIZE = (128, 128)


class ImageDatasetSequence(Sequence):
    """Represents a single sequence in the dataset (i.e., the train,
    val, or test set)."""

    def __init__(self, x_filenames: np.ndarray, y: Optional[np.ndarray] = None,
                 image_target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
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
        :param image_target_size: the size at which images will be loaded.
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
        self.image_target_size: Tuple[int, int] = image_target_size
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
            if SHUFFLE_ON_EPOCH_END:
                raise ValueError('Cannot overfit on one batch if shuffling is '
                                 'true.')
            idx = 0
        batch_start = idx * self.batch_size
        if batch_start >= self.x_filenames.shape[0] - 1:
            raise ValueError('Batch starting index too large for dataset.')
        batch_end = min((idx + 1) * self.batch_size, self.x_filenames.shape[0])
        batch_x_filenames = self.x_filenames[batch_start:batch_end]
        batch_x = np.array([img_to_array(load_img(
            filename, target_size=self.image_target_size), dtype=np.uint8)
            for filename in batch_x_filenames])
        batch_x = normalize_images(batch_x)
        batch = batch_x, self.y[batch_start:batch_end]
        if self.batch_augment_fn:
            batch = self.batch_augment_fn(batch)
        if self.batch_format_fn:
            batch = self.batch_format_fn(batch)
        return batch

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
