"""Tests the ilsvrc_dataset module."""

import pytest
import os

from dataset.ilsvrc_dataset import ILSVRCDataset, DEFAULT_DATASET_PATH, \
    EXPECTED_NUM_CLASSES, TRAIN_KEY, VAL_KEY, TEST_KEY


@pytest.fixture
def dataset() -> ILSVRCDataset:
    """Returns an ILSVRCDataset."""
    return ILSVRCDataset(DEFAULT_DATASET_PATH)


def test_label_mapping(dataset) -> None:
    """Tests that the ILSVRCDataset's label mapping is correct. In
    particular, there should be 1000 classes with descriptive class
    names; synset IDs should correspond to correct classnames; and
    classnames/labels should be correctly associated."""
    assert len(dataset.label_to_classname) == EXPECTED_NUM_CLASSES
    assert len(dataset.classname_to_label.keys()) == EXPECTED_NUM_CLASSES
    assert len(dataset.synid_to_classname.keys()) == EXPECTED_NUM_CLASSES
    assert dataset.synid_to_classname['n01484850'] == \
        'great white shark, white shark, man-eater, man-eating shark, ' \
        'Carcharodon carcharias'
    assert dataset.synid_to_classname['n02508021'] == 'raccoon, racoon'
    assert dataset.synid_to_classname['n06267145'] == 'newspaper, paper'
    assert dataset.label_to_classname[0] == \
        'great white shark, white shark, man-eater, man-eating shark, ' \
        'Carcharodon carcharias'
    assert dataset.label_to_classname[999] == 'bean'
    assert dataset.label_to_classname[995] == 'bolete'
    assert dataset.label_to_classname[10] == 'bullfrog, Rana catesbeiana'
    for label_idx, classname in enumerate(dataset.label_to_classname):
        assert dataset.classname_to_label[classname] == label_idx


def test_partition(dataset) -> None:
    """Tests that ILSVRCDataset's partition is filled correctly. In
    particular, the filepaths should point to the correct files and
    the train/val labels should be correct."""
    # Train.
    fname = os.path.join(dataset.path, 'train', 'n01807496',
                          'n01807496_8.JPEG')
    assert fname in dataset.partition[TRAIN_KEY]
    assert fname not in dataset.partition[VAL_KEY]
    assert fname not in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    assert fname in dataset._labels
    assert dataset._labels[fname] == dataset.classname_to_label[
        dataset.synid_to_classname['n01807496']]
    fname = os.path.join(dataset.path, 'train', 'n04118538',
                         'n04118538_570.JPEG')
    assert fname in dataset.partition[TRAIN_KEY]
    assert fname not in dataset.partition[VAL_KEY]
    assert fname not in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    assert fname in dataset._labels
    assert dataset._labels[fname] == dataset.classname_to_label[
        dataset.synid_to_classname['n04118538']]
    # Val.
    fname = os.path.join(dataset.path, 'val', 'ILSVRC2010_val_00000001.JPEG')
    assert fname not in dataset.partition[TRAIN_KEY]
    assert fname in dataset.partition[VAL_KEY]
    assert fname not in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    assert fname in dataset._labels
    assert dataset._labels[fname] == 78
    assert dataset.classname_to_label[dataset.synid_to_classname['n09428293']] == 78
    assert dataset.label_to_classname[78] == 'TODO what is this label?'
    fname = os.path.join(dataset.path, 'val', 'ILSVRC2010_val_00008079.JPEG')
    assert fname not in dataset.partition[TRAIN_KEY]
    assert fname in dataset.partition[VAL_KEY]
    assert fname not in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    fname = os.path.join(dataset.path, 'val', 'ILSVRC2010_val_00050000.JPEG')
    assert fname not in dataset.partition[TRAIN_KEY]
    assert fname in dataset.partition[VAL_KEY]
    assert fname not in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    # Test.
    fname = os.path.join(dataset.path, 'test', 'ILSVRC2010_test_00015978.JPEG')
    assert fname not in dataset.partition[TRAIN_KEY]
    assert fname not in dataset.partition[VAL_KEY]
    assert fname in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    assert fname not in dataset._labels
    fname = os.path.join(dataset.path, 'test', 'ILSVRC2010_test_00150000.JPEG')
    assert fname not in dataset.partition[TRAIN_KEY]
    assert fname not in dataset.partition[VAL_KEY]
    assert fname in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    assert fname not in dataset._labels
