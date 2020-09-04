"""Dataset class."""

import os
from typing import Dict, List
import numpy as np

VERBOSE: bool = True
TRAIN_KEY: str = 'train'
VAL_KEY: str = 'val'
TEST_KEY: str = 'test'
EMPTY_PARTITION: Dict[str, np.ndarray] = {
    TRAIN_KEY: np.arange(0, dtype='str'),
    VAL_KEY: np.arange(0, dtype='str'),
    TEST_KEY: np.arange(0, dtype='str')
}


class Dataset:
    """Represents a dataset."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.
        :param path: the path to the dataset if it exists, or the path
        to which the root directory should be saved.
        """
        self.path: str = path
        self.partition: Dict[str, np.ndarray] = EMPTY_PARTITION
        self.labels: Dict[str, int] = {}
        self.class_mapping: List[str] = []
        self.inverse_mapping: Dict[str, int] = {}
        self._load_or_download_dataset(verbose=VERBOSE)

    def get_labels(self, x_filenames: np.ndarray) -> np.ndarray:
        """Returns an np.ndarray of ints representing the labels for
        the given filenames.
        :param x_filenames: the names of the training files.
        :return: the labels, in the same order as the filenames.
        """
        return np.array([self.labels[fname] for fname in x_filenames])

    def _load_or_download_dataset(self, verbose: bool = False) -> None:
        """Loads the dataset from a pre-existing path, or downloads it.
        Subclasses should override.
        :param verbose: whether to print info to stdout.
        """
        if not os.path.exists(self.path):
            if verbose:
                print('Making dir {0}'.format(self.path))
            os.mkdir(self.path)
        if not os.path.isdir(self.path):
            raise NotADirectoryError('{0} is not a directory.'.format(
                self.path))
