"""Tests the train_model module."""

import pytest

from training import train_model

NUM_CLASSES = 1000
OVERFIT_SINGLE_BATCH_TARGET_LOSS = 0.001


def test_get_model() -> None:
    """Tests that the model is receiving parameters and getting created
    correctly."""
    dataset_args = {'dataset_fraction': 0.01}
    network_args = {'input_shape': (64, 64, 3),
                    'num_classes': 1000}
    model = train_model.get_model(dataset_args, network_args)
    assert model.network.layers[0].output_shape[1] == (64 * 64 * 3)
    assert model.network.layers[-1].output_shape[1] == 1000
    dataset_args = {'dataset_fraction': 0.001}
    network_args = {'input_shape': (32, 32, 3),
                    'num_classes': 999}
    model = train_model.get_model(dataset_args, network_args)
    assert model.network.layers[0].output_shape[1] == (32 * 32 * 3)
    assert model.network.layers[-1].output_shape[1] == 999


def test_train_model() -> None:
    """Tests that the model is being trained correctly."""
    dataset_args = {'dataset_fraction': 0.1}
    network_args = {'input_shape': (32, 32, 3),
                    'num_classes': NUM_CLASSES - 1}
    train_args = {}
    model = train_model.get_model(dataset_args, network_args)
    assert model.network.layers[0].output_shape[1] == (32 * 32 * 3)
    assert model.network.layers[-1].output_shape[1] == NUM_CLASSES - 1
    # The output shape is lower than the number of classes, so training fails.
    with pytest.raises(IndexError):
        train_model.train_model(model, train_args)
    dataset_args = {'dataset_fraction': 0.01}
    network_args = {'input_shape': (32, 32, 3),
                    'num_classes': NUM_CLASSES}
    train_args = {}
    model = train_model.get_model(dataset_args, network_args)
    assert model.network.layers[0].output_shape[1] == (32 * 32 * 3)
    assert model.network.layers[-1].output_shape[1] == NUM_CLASSES
    history = train_model.train_model(model, train_args)
    loss_by_epoch = history.history['loss']
    acc_by_epoch = history.history['accuracy']
    # Test that training improved loss and accuracy.
    assert loss_by_epoch[0] > loss_by_epoch[-1]
    assert acc_by_epoch[0] < acc_by_epoch[-1]


def test_overfit_single_batch() -> None:
    """Tests that the model can overfit on a single batch."""
    dataset_args = {'dataset_fraction': 0.01}
    network_args = {'input_shape': (32, 32, 3),
                    'num_classes': NUM_CLASSES}
    train_args = {
        'batch_size': 32,
        'epochs': 30,
        'early_stopping': False,
        'overfit_single_batch': True,
        'shuffle_on_epoch_end': False
    }
    model = train_model.get_model(dataset_args, network_args)
    history = train_model.train_model(model, train_args)
    loss_by_epoch = history.history['loss']
    acc_by_epoch = history.history['accuracy']
    # Test that training improved loss and accuracy.
    assert loss_by_epoch[0] > loss_by_epoch[-1]
    assert acc_by_epoch[0] < acc_by_epoch[-1]
    # Test that we can get training loss arbitrarily low.
    assert loss_by_epoch[-1] < OVERFIT_SINGLE_BATCH_TARGET_LOSS
