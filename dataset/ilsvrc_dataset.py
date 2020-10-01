"""ILSVRCDataset class."""

import os
import numpy as np
from typing import Dict, List

from dataset.dataset import Dataset, TRAIN_KEY, VAL_KEY, TEST_KEY
from dataset.image_dataset_sequence import ImageDatasetSequence

DEFAULT_DATASET_PATH = os.path.join(
    '/',
    'Users',
    'leo',
    'Documents',
    'Datasets',
    'ILSVRC2012'
)
CLASS_MAPPING_DIR = 'class_mapping'
WORDS_FILENAME = os.path.join(CLASS_MAPPING_DIR, 'words.txt')
WNIDS_FILENAME = os.path.join(CLASS_MAPPING_DIR, 'wnids.txt')
VAL_LABELS_FILE = os.path.join(
    'devkit-1.0',
    'data',
    'ILSVRC2010_validation_ground_truth.txt'
)
EXPECTED_NUM_CLASSES = 1000
NULL_LABEL = 'NULL'


class ILSVRCDataset(Dataset):
    """Represents the ILSVRC2012 dataset."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.
        :param path: the path to the dataset if it exists, or the path
        to which the root directory should be saved.
        """
        self.path_train: str = os.path.join(path, 'train')
        self.path_val: str = os.path.join(path, 'val')
        self.path_test: str = os.path.join(path, 'test')
        self.synid_to_classname: Dict[str, str] = {}
        super().__init__(path)

    def _get_train_dirs(self) -> List[str]:
        """Returns a list of the training directories.
        :return: the training directories, unsorted.
        """
        return [dirname for dirname in os.listdir(self.path_train)
                if os.path.isdir(os.path.join(self.path_train, dirname))]

    def _get_synids(self) -> List[str]:
        """Returns the synset ID values by which classes are
        identified. For example, 'n01484850' is a great white shark.
        Because this is how the classes are determined for the training
        images, this is the same as _get_train_dirs().
        :return: the synset ID values of all classes, unsorted.
        """
        return self._get_train_dirs()

    def _get_full_synid_mapping(self) -> Dict[str, str]:
        """Returns the complete mapping from synset ID to class name.
        This is found in the WORDS_FILENAME file. Only the 1000 IDs
        used in the dataset are saved to synid_to_classname; this is
        just a helper method to populate that dict.
        :return: a dict where the keys are synset IDs and the values
        are class names.
        """
        full_mapping = {}
        with open(os.path.join(self.path, WORDS_FILENAME)) as infile:
            for line in infile.readlines():
                pair = line.split('\t')
                if len(pair) != 2:
                    raise ValueError('Expected exactly one tab per line, but '
                                     'found {0}.'.format(len(pair) - 1))
                synid, classname = pair[0].strip(), pair[1].strip()
                full_mapping[synid] = classname
        return full_mapping

    def _get_wnid_to_label(self) -> Dict[str, int]:
        """Returns the dict mapping synset ID (also WNID) to label
        number. Fixed ImageNet's scheme to be zero-indexed.
        :return: the WNID/synset ID to label dict.
        """
        wnid_to_label = {}
        with open(os.path.join(self.path, WNIDS_FILENAME)) as infile:
            lines = infile.readlines()
            for idx, line in enumerate(lines):
                wnid = line.split()[2].strip()
                wnid_to_label[wnid] = idx
        return wnid_to_label

    def _fill_label_mapping(self) -> None:
        """Fills self.label_to_classname, self.classname_to_label, and
        self._synid_to_classname."""
        full_mapping = self._get_full_synid_mapping()
        used_synids = self._get_synids()
        wnid_to_label = self._get_wnid_to_label()
        self.label_to_classname = [NULL_LABEL for _ in range(
            EXPECTED_NUM_CLASSES)]
        for i, synid in enumerate(used_synids):
            classname = full_mapping[synid]
            label = wnid_to_label[synid]
            self.synid_to_classname[synid] = classname
            self.label_to_classname[label] = classname
            if classname in self.classname_to_label:
                raise ValueError(
                    'Class {0} already assigned to label {1}'.format(
                        classname, self.classname_to_label[classname]))
            self.classname_to_label[classname] = label

    def _fill_partition(self) -> None:
        """Fills self.partition and self._labels."""
        train_images = []
        train_dirs = self._get_train_dirs()
        for class_dir in train_dirs:
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
            val_labels = [int(line.strip()) - 1 for line in infile.readlines()]
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
