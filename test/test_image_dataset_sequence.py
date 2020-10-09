"""Tests the image_dataset_sequence module."""

import pytest
from dataset.image_dataset_sequence import ImageDatasetSequence
from dataset.ilsvrc_dataset import ILSVRCDataset, DEFAULT_DATASET_PATH, \
    TRAIN_KEY

DATASET_FRACTION = 0.01
NUM_CLASSES = 1000
IMAGE_TARGET_SIZE = (128, 128)
BATCH_SIZE = 32


@pytest.fixture
def dataset_small() -> ILSVRCDataset:
    """Returns an ILSVRCDataset."""
    dataset = ILSVRCDataset(DEFAULT_DATASET_PATH)
    dataset.trim_dataset(DATASET_FRACTION)
    return dataset


def test_images() -> None:
    """Tests that the sequence output images meet expected standards."""
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



def test_shuffle() -> None:
    """Tests that the shuffling flag works as expected."""
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
