"""Dataset class."""

import os
from typing import Dict, List
import numpy as np
from tensorflow.keras.utils import to_categorical

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
        self._labels: Dict[str, int] = {}
        self.label_to_classname: List[str] = []
        self.classname_to_label: Dict[str, int] = {}
        self._load_or_download_dataset(verbose=VERBOSE)

    def get_labels(self, x_filenames: np.ndarray, categorical: bool,
                   categorical_num_classes: int) -> np.ndarray:
        """Returns an np.ndarray of ints representing the labels for
        the given filenames.
        :param x_filenames: the names of the training files.
        :param categorical: whether to return the labels as categorical values.
        :param categorical_num_classes: the number of classes in the dataset;
        only used if categorical is True.
        :return: the labels, in the same order as the filenames.
        """
        arr = np.array([self._labels[fname] for fname in x_filenames])
        if categorical:
            return to_categorical(arr, num_classes=categorical_num_classes)
        return arr

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

    def trim_dataset(self, target_dataset_fraction: float,
                     trim_train: bool = True, trim_val: bool = True,
                     trim_test: bool = False) -> None:
        """Reduces the size of the dataset (train, val, and/or test) so
        that only target_dataset_fraction of them are used.
        :param target_dataset_fraction: the fraction of data to keep,
        in the interval [0.0, 1.0].
        :param trim_train: whether to trim the training set.
        :param trim_val: whether to trim the validation set.
        :param trim_test: whether to trim the test set.
        """
        if trim_train:
            end_idx = int(self.partition[TRAIN_KEY].shape[0]
                          * target_dataset_fraction)
            self.partition[TRAIN_KEY] = self.partition[TRAIN_KEY][:end_idx]
        if trim_val:
            end_idx = int(self.partition[VAL_KEY].shape[0]
                          * target_dataset_fraction)
            self.partition[VAL_KEY] = self.partition[VAL_KEY][:end_idx]
        if trim_test:
            end_idx = int(self.partition[TEST_KEY].shape[0]
                          * target_dataset_fraction)
            self.partition[TEST_KEY] = self.partition[TEST_KEY][:end_idx]
