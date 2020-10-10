"""Tests the image_dataset_sequence module."""

import pytest
import numpy as np

from dataset.image_dataset_sequence import ImageDatasetSequence
from dataset.ilsvrc_dataset import ILSVRCDataset, DEFAULT_DATASET_PATH, \
    TRAIN_KEY

DATASET_FRACTION = 0.001
NUM_CLASSES = 1000
IMAGE_TARGET_SIZE = (128, 128)
BATCH_SIZE = 32


@pytest.fixture
def dataset() -> ILSVRCDataset:
    """Returns an ILSVRCDataset."""
    dataset = ILSVRCDataset(DEFAULT_DATASET_PATH)
    return dataset


def test_images(dataset: ILSVRCDataset) -> None:
    """Tests that the sequence output images meet expected standards."""
    dataset.trim_dataset(DATASET_FRACTION)
    x_train_filenames = dataset.partition[TRAIN_KEY]
    y_train = dataset.get_labels(x_train_filenames, True, NUM_CLASSES)
    train_sequence = ImageDatasetSequence(
        x_train_filenames, y=y_train, batch_size=BATCH_SIZE,
        image_target_size=IMAGE_TARGET_SIZE,
        batch_augment_fn=None,
        batch_format_fn=None,
        overfit_single_batch=False,
        shuffle_on_batch_end=True
    )
    # Test that only the last batch is not of length BATCH_SIZE.
    # Also test that there are the correct number of batches.
    on_last_batch = False
    num_batches_seen = 0
    for batch in train_sequence:
        assert not on_last_batch
        x_batch, y_batch = batch
        # Take the first image/label pair and check that it meets standards.
        # Check that the image is of the right size.
        assert x_batch[0].shape == IMAGE_TARGET_SIZE + (3,)
        # Check that the image is of the right datatype.
        assert x_batch.dtype == np.float32
        # Check that the label is categorical and of the right dimension.
        assert y_batch.shape[1] == NUM_CLASSES
        # Check that the label is of the right datatype.
        assert y_batch.dtype == np.float32
        on_last_batch = not (x_batch.shape[0] == BATCH_SIZE and
                             y_batch.shape[0] == BATCH_SIZE)
        num_batches_seen += 1
    assert num_batches_seen == len(train_sequence)


def test_shuffle() -> None:
    """Tests that the shuffling flag works as expected. Also tests that
    filenames and labels are still properly mapped."""
    x_train_filenames = dataset.partition[TRAIN_KEY]
    y_train = dataset.get_labels(x_train_filenames, True, NUM_CLASSES)
    train_sequence = ImageDatasetSequence(
        x_train_filenames, y=y_train, batch_size=BATCH_SIZE,
        image_target_size=IMAGE_TARGET_SIZE,
        batch_augment_fn=None,
        batch_format_fn=None,
        overfit_single_batch=False,
        shuffle_on_batch_end=True
    )
    # TODO
