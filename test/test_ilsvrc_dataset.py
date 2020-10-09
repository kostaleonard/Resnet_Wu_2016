"""Tests the ilsvrc_dataset module."""

import pytest
import os

from dataset.ilsvrc_dataset import ILSVRCDataset, DEFAULT_DATASET_PATH, \
    EXPECTED_NUM_CLASSES, TRAIN_KEY, VAL_KEY, TEST_KEY, NULL_LABEL

DATASET_FRACTION = 0.1
DELTA = 0.05


@pytest.fixture
def dataset() -> ILSVRCDataset:
    """Returns an ILSVRCDataset."""
    return ILSVRCDataset(DEFAULT_DATASET_PATH)


def test_trim_dataset(dataset: ILSVRCDataset) -> None:
    """Tests that the dataset is being trimmed properly. The trimmed
    dataset should be shuffled so that the classes retain the same
    approximate distribution.
    :param dataset: the dataset.
    """
    train_size_before = dataset.partition[TRAIN_KEY].shape[0]
    val_size_before = dataset.partition[VAL_KEY].shape[0]
    test_size_before = dataset.partition[TEST_KEY].shape[0]
    train_subset_before = dataset.partition[TRAIN_KEY][:5]
    val_subset_before = dataset.partition[VAL_KEY][:5]
    test_subset_before = dataset.partition[TEST_KEY][:5]
    dataset.trim_dataset(DATASET_FRACTION, trim_val=True, trim_test=False)
    train_size_after = dataset.partition[TRAIN_KEY].shape[0]
    val_size_after = dataset.partition[VAL_KEY].shape[0]
    test_size_after = dataset.partition[TEST_KEY].shape[0]
    train_subset_after = dataset.partition[TRAIN_KEY][:5]
    val_subset_after = dataset.partition[VAL_KEY][:5]
    test_subset_after = dataset.partition[TEST_KEY][:5]
    # Check that trimming occurred (or didn't).
    assert (train_size_before * (DATASET_FRACTION - DELTA)) < \
        train_size_after < \
        (train_size_before * (DATASET_FRACTION + DELTA))
    assert (val_size_before * (DATASET_FRACTION - DELTA)) < \
        val_size_after < \
        (val_size_before * (DATASET_FRACTION + DELTA))
    assert test_size_before == test_size_after
    # Check that the datasets were shuffled (or weren't).
    # We're just going to use the first 5 filenames to check for shuffling;
    # it's extremely unlikely that all are the same after shuffling.
    assert (train_subset_before != train_subset_after).any()
    assert (val_subset_before != val_subset_after).any()
    assert (test_subset_before == test_subset_after).all()


def test_label_mapping(dataset: ILSVRCDataset) -> None:
    """Tests that the ILSVRCDataset's label mapping is correct. In
    particular, there should be 1000 classes with descriptive class
    names; synset IDs should correspond to correct classnames; and
    classnames/labels should be correctly associated.
    :param dataset: the dataset.
    """
    assert len(dataset.label_to_classname) == EXPECTED_NUM_CLASSES
    assert len(dataset.classname_to_label.keys()) == EXPECTED_NUM_CLASSES
    assert len(dataset.synid_to_classname.keys()) == EXPECTED_NUM_CLASSES
    assert dataset.synid_to_classname['n01484850'] == \
        'great white shark, white shark, man-eater, man-eating shark, ' \
        'Carcharodon carcharias'
    assert dataset.synid_to_classname['n02508021'] == 'raccoon, racoon'
    assert dataset.synid_to_classname['n06267145'] == 'newspaper, paper'
    assert dataset.label_to_classname[0] == \
        'french fries, french-fried potatoes, fries, chips'
    assert dataset.label_to_classname[10] == 'blackberry'
    assert dataset.label_to_classname[999] == 'washer, automatic washer, ' \
                                              'washing machine'
    for label_idx, classname in enumerate(dataset.label_to_classname):
        assert dataset.classname_to_label[classname] == label_idx


def test_partition(dataset: ILSVRCDataset) -> None:
    """Tests that ILSVRCDataset's partition is filled correctly. In
    particular, the filepaths should point to the correct files and
    the train/val labels should be correct.
    :param dataset: the dataset.
    """
    assert NULL_LABEL not in dataset.label_to_classname
    # Train.
    fname = os.path.join(dataset.path, 'train', 'n01807496',
                         'n01807496_8.JPEG')
    assert fname in dataset.partition[TRAIN_KEY]
    assert fname not in dataset.partition[VAL_KEY]
    assert fname not in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    assert fname in dataset._labels
    assert dataset.synid_to_classname['n01807496'] == 'partridge'
    assert dataset._labels[fname] == dataset.classname_to_label[
        dataset.synid_to_classname['n01807496']]
    assert dataset._labels[fname] == 390
    fname = os.path.join(dataset.path, 'train', 'n04118538',
                         'n04118538_570.JPEG')
    assert fname in dataset.partition[TRAIN_KEY]
    assert fname not in dataset.partition[VAL_KEY]
    assert fname not in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    assert fname in dataset._labels
    assert dataset.synid_to_classname['n04118538'] == 'rugby ball'
    assert dataset._labels[fname] == dataset.classname_to_label[
        dataset.synid_to_classname['n04118538']]
    assert dataset._labels[fname] == 749
    # Val.
    fname = os.path.join(dataset.path, 'val', 'ILSVRC2010_val_00000001.JPEG')
    assert fname not in dataset.partition[TRAIN_KEY]
    assert fname in dataset.partition[VAL_KEY]
    assert fname not in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    assert fname in dataset._labels
    assert dataset._labels[fname] == 77
    assert dataset.classname_to_label[
               dataset.synid_to_classname['n09428293']] == 77
    assert dataset.label_to_classname[77] == \
           'seashore, coast, seacoast, sea-coast'
    fname = os.path.join(dataset.path, 'val', 'ILSVRC2010_val_00008079.JPEG')
    assert fname not in dataset.partition[TRAIN_KEY]
    assert fname in dataset.partition[VAL_KEY]
    assert fname not in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    assert dataset._labels[fname] == 734
    assert dataset.classname_to_label[
               dataset.synid_to_classname['n03535780']] == 734
    assert dataset.label_to_classname[734] == 'horizontal bar, high bar'
    fname = os.path.join(dataset.path, 'val', 'ILSVRC2010_val_00050000.JPEG')
    assert fname not in dataset.partition[TRAIN_KEY]
    assert fname in dataset.partition[VAL_KEY]
    assert fname not in dataset.partition[TEST_KEY]
    assert os.path.exists(fname)
    assert os.path.isfile(fname)
    assert dataset._labels[fname] == 560
    assert dataset.classname_to_label[
               dataset.synid_to_classname['n04090263']] == 560
    assert dataset.label_to_classname[560] == 'rifle'
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
