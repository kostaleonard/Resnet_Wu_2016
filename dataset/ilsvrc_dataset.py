"""ILSVRCDataset class."""

import os
import numpy as np
from typing import Dict

from dataset.dataset import Dataset, TRAIN_KEY, VAL_KEY, TEST_KEY

CLASS_MAPPING_DIR = 'class_mapping'
WORDS_FILENAME = 'words.txt'
VAL_LABELS_FILE = os.path.join(
    'devkit-1.0',
    'data',
    'ILSVRC2010_validation_ground_truth.txt'
)


class ILSVRCDataset(Dataset):
    """Represents the ILSVRC2012 dataset."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.
        :param path: the path to the dataset if it exists, or the path
        to which the root directory should be saved.
        """
        self.path_train: str = os.path.join(self.path, 'train')
        self.path_val: str = os.path.join(self.path, 'val')
        self.path_test: str = os.path.join(self.path, 'test')
        self.synid_to_classname: Dict[str, str] = {}
        super().__init__(path)

    def _fill_label_mapping(self):
        """Fills self.label_to_classname, self.classname_to_label, and
        self._synid_to_classname."""
        sorted_synids = sorted(os.listdir(self.path_train))
        with open(os.path.join(self.path, CLASS_MAPPING_DIR, WORDS_FILENAME)) \
                as infile:
            full_mapping = {}
            for line in infile.readlines():
                pair = line.split('\t')
                if len(pair) != 2:
                    raise ValueError('Expected exactly one tab per line, but '
                                     'found {0}.'.format(len(pair) - 1))
                synid, classname = pair[0].strip(), pair[1].strip()
                full_mapping[synid] = classname
            for synid in sorted_synids:
                classname = full_mapping[synid]
                self.synid_to_classname[synid] = classname
                self.label_to_classname.append(classname)
                self.classname_to_label[classname] = \
                    len(self.label_to_classname) - 1

    def _fill_partition(self):
        """Fills self.partition and self._labels."""
        train_images = []
        for class_dir in os.listdir(self.path_train):
            class_path = os.path.join(self.path_train, class_dir)
            for imfile in os.listdir(class_path):
                imfile_path = os.path.join(class_path, imfile)
                train_images.append(str(imfile_path))
                synid = str(imfile).split('_')[0]
                label = self.classname_to_label[self.synid_to_classname[synid]]
                self._labels[imfile_path] = label
        self.partition[TRAIN_KEY] = np.array(train_images, dtype='str')
        val_images = []
        with open(os.path.join(self.path, VAL_LABELS_FILE)) as infile:
            val_labels = [int(line.strip()) for line in infile.readlines()]
            val_imfiles = sorted(os.listdir(self.path_val))
            if len(val_labels) != len(val_imfiles):
                raise ValueError(
                    'Expected the same number of labels and images, but found '
                    '{0} labels, {1} images.'.format(
                        len(val_labels), len(val_imfiles)))
            for i, imfile in enumerate(val_imfiles):
                imfile_path = os.path.join(self.path_val, imfile)
                val_images.append(imfile_path)
                self._labels[imfile_path] = val_labels[i]
        self.partition[VAL_KEY] = np.array(val_images, dtype='str')
        test_images = []
        for imfile in os.listdir(self.path_test):
            imfile_path = os.path.join(self.path_test, imfile)
            test_images.append(imfile_path)
        self.partition[TEST_KEY] = np.array(test_images, dtype='str')

    def _load_or_download_dataset(self, verbose: bool = False) -> None:
        """Loads the dataset from a pre-existing path, or downloads it.
        :param verbose: whether to print info to stdout.
        """
        super()._load_or_download_dataset(verbose=verbose)
        self._fill_label_mapping()
        self._fill_partition()
