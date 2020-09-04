"""ILSVRCDataset class."""

import os
import numpy as np

from dataset.dataset import Dataset, TRAIN_KEY, VAL_KEY, TEST_KEY


class ILSVRCDataset(Dataset):
    """Represents the ILSVRC2012 dataset."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.
        :param path: the path to the dataset if it exists, or the path
        to which the root directory should be saved.
        """
        self.path_train = os.path.join(self.path, 'train')
        self.path_val = os.path.join(self.path, 'val')
        self.path_test = os.path.join(self.path, 'test')
        super().__init__(path)

    def _load_or_download_dataset(self, verbose: bool = False) -> None:
        """Loads the dataset from a pre-existing path, or downloads it.
        :param verbose: whether to print info to stdout.
        """
        super()._load_or_download_dataset(verbose=verbose)
        train_images = []
        for class_dir in os.listdir(self.path_train):
            class_path = os.path.join(self.path_train, class_dir)
            for imfile in os.listdir(class_path):
                imfile_path = os.path.join(class_path, imfile)
                train_images.append(imfile_path)
                # TODO labels for each class in self.labels
        self.partition[TRAIN_KEY] = np.array(train_images, dtype='str')
        val_images = []
        for imfile in os.listdir(self.path_val):
            imfile_path = os.path.join(self.path_val, imfile)
            val_images.append(imfile_path)
            # TODO labels in file
        self.partition[VAL_KEY] = np.array(val_images, dtype='str')
        test_images = []
        for imfile in os.listdir(self.path_test):
            imfile_path = os.path.join(self.path_test, imfile)
            test_images.append(imfile_path)
        self.partition[TEST_KEY] = np.array(test_images, dtype='str')
        # TODO class mapping and inverse mapping

