"""Utility functions."""

import numpy as np


def get_max_batch_size(x_filenames: np.ndarray) -> int:
    """Returns the maximum batch size attainable on the current
    CPU/GPU.
    :param x_filenames: ndarray of strs containing the filenames of all
    examples in the dataset.
    :return: maximum batch size.
    """
    # TODO
