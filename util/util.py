"""Utility functions."""

# Random seeds need to be set up at program launch, before other
# imports, because some libraries use random initialization.
from numpy.random import seed
import numpy as np

RANDOM_SEED = 52017


def set_random_seed(random_seed: int = RANDOM_SEED) -> None:
    """Sets up the random seed so that experiments are reproducible.
    :param random_seed: the random seed to use.
    """
    seed(random_seed)


def get_max_batch_size(x_filenames: np.ndarray) -> int:
    """Returns the maximum batch size attainable on the current
    CPU/GPU.
    :param x_filenames: ndarray of strs containing the filenames of all
    examples in the dataset.
    :return: maximum batch size.
    """
    # TODO


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Returns a normalized data array for one or more images.
    :param images: an np.ndarray of one or more images.
    :return: a normalized array.
    """
    if images.dtype != np.uint8:
        raise ValueError('Expected processed images to be of data '
                         'type np.uint8, but found {0}'.format(images.dtype))
    return (images / 255).astype(np.float32)
