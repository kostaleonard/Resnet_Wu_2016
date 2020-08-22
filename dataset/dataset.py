"""Contains the Dataset class."""

import os
from typing import Dict, List

VERBOSE: bool = True
TRAIN_KEY: str = 'train'
VAL_KEY: str = 'val'
TEST_KEY: str = 'test'
EMPTY_PARTITION: Dict[str, List[str]] = {
    TRAIN_KEY: [],
    VAL_KEY: [],
    TEST_KEY: []
}


class Dataset:
    """Represents a dataset."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.
        :param path: the path to the dataset if it exists, or the path
        to which the root directory should be saved.
        """
        self.path: str = path
        self.partition: Dict[str, List[str]] = EMPTY_PARTITION
        self.labels: Dict[str, int] = {}
        self._load_or_download_dataset(verbose=VERBOSE)

    def _load_or_download_dataset(self, verbose: bool = False) -> None:
        """Loads the dataset from a pre-existing path, or downloads it.
        :param verbose: whether to print info to stdout.
        """
        if not os.path.exists(self.path):
            if verbose:
                print('Making dir {0}'.format(self.path))
            os.mkdir(self.path)
        if not os.path.isdir(self.path):
            raise NotADirectoryError('{0} is not a directory.'.format(
                self.path))
