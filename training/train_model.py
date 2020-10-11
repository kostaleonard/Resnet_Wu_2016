"""Trains the model."""

from util.util import set_random_seed
USE_RANDOM_SEED = True
if USE_RANDOM_SEED:
    set_random_seed()
# pylint: disable=wrong-import-position
from tensorflow.keras.callbacks import EarlyStopping, Callback, History
from wandb.keras import WandbCallback
from typing import List, Dict, Any
from time import time

from models.project_model import ProjectModel, DEFAULT_TRAIN_ARGS
from models.image_model import ImageModel
from dataset.ilsvrc_dataset import ILSVRCDataset, DEFAULT_DATASET_PATH, \
    EXPECTED_NUM_CLASSES
from dataset.image_dataset_sequence import DEFAULT_TARGET_SIZE
from dataset.dataset import TRAIN_KEY, VAL_KEY, TEST_KEY
from models.networks.mlp import MLP
from models.networks.lenet import LeNet

ARCHITECTURE_MLP = 'mlp'
ARCHITECTURE_LENET = 'lenet'
DEFAULT_DATASET_ARGS = {
    'dataset_fraction': 0.01
}
DEFAULT_NETWORK_ARGS = {
    'architecture': ARCHITECTURE_MLP,
    'input_shape': DEFAULT_TARGET_SIZE + (3,),
    'num_classes': EXPECTED_NUM_CLASSES
}


def get_custom_wandb_callbacks() -> List[Callback]:
    """Returns a list of custom wandb callbacks to use.
    :return: custom callbacks.
    """
    # TODO custom callbacks.
    return []


def get_model(dataset_args: Dict[str, Any],
              network_args: Dict[str, Any]) -> ProjectModel:
    """Returns the model.
    :param dataset_args: the dataset arguments; see DEFAULT_DATASET_ARGS for
    available arguments.
    :param network_args: the network arguments; see DEFAULT_NETWORK_ARGS for
    available arguments.
    :return: the model.
    """
    dataset_args = {**DEFAULT_DATASET_ARGS, **dataset_args}
    network_args = {**DEFAULT_NETWORK_ARGS, **network_args}
    print('Dataset args: {0}'.format(dataset_args))
    print('Network args: {0}'.format(network_args))
    print('Loading dataset from {0}'.format(DEFAULT_DATASET_PATH))
    dataset = ILSVRCDataset(DEFAULT_DATASET_PATH)
    if dataset_args['dataset_fraction'] < 1.0:
        dataset.trim_dataset(dataset_args['dataset_fraction'])
    print('Num training examples: {0}'.format(
        dataset.partition[TRAIN_KEY].shape[0]))
    print('Num validation examples: {0}'.format(
        dataset.partition[VAL_KEY].shape[0]))
    print('Num test examples: {0}'.format(
        dataset.partition[TEST_KEY].shape[0]))
    if network_args['architecture'] == ARCHITECTURE_MLP:
        network = MLP(network_args)
    elif network_args['architecture'] == ARCHITECTURE_LENET:
        network = LeNet(network_args)
    else:
        raise ValueError('Unrecognized architecture: {0}'.format(
            network_args['architecture']))
    return ImageModel(dataset, network)


def train_model(model: ProjectModel, train_args: Dict[str, Any],
                use_wandb: bool = False) -> History:
    """Trains the model.
    :param model: the model to train.
    :param train_args: training arguments; see DEFAULT_TRAIN_ARGS for
    available arguments.
    :param use_wandb: whether to sync the training run to wandb.
    :return: the training history.
    """
    train_args = {**DEFAULT_TRAIN_ARGS, **train_args}
    print('Training args: {0}'.format(train_args))
    callbacks = []
    if train_args['early_stopping']:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01,
                                       patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)
    if use_wandb:
        callbacks.append(WandbCallback())
        callbacks.extend(get_custom_wandb_callbacks())
    model.network.summary()
    t_start = time()
    history = model.fit(train_args, callbacks=callbacks)
    t_end = time()
    print('Model training finished in {0:2f}s'.format(t_end - t_start))
    return history


def main() -> None:
    """Runs the program."""
    # TODO actually use dataset_args, network_args
    model = get_model({}, {'architecture': ARCHITECTURE_LENET})
    # TODO actually use train_args
    history = train_model(model, {})
    print(history.history)


if __name__ == '__main__':
    main()
