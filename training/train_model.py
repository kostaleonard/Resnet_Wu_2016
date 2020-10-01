"""Trains the model."""

from util.util import set_random_seed
USE_RANDOM_SEED = True
if USE_RANDOM_SEED:
    set_random_seed()
# pylint: disable=wrong-import-position
from tensorflow.keras.callbacks import EarlyStopping, Callback
from wandb.keras import WandbCallback
from typing import List, Dict, Any
from time import time

from models.project_model import ProjectModel, DEFAULT_TRAIN_ARGS
from dataset.ilsvrc_dataset import ILSVRCDataset, DEFAULT_DATASET_PATH, \
    EXPECTED_NUM_CLASSES
from dataset.image_dataset_sequence import DEFAULT_TARGET_SIZE
from models.networks.mlp import MLP

DEFAULT_NETWORK_ARGS = {
    'input_shape': DEFAULT_TARGET_SIZE + (3,),
    'num_classes': EXPECTED_NUM_CLASSES
}


def get_custom_wandb_callbacks() -> List[Callback]:
    """Returns a list of custom wandb callbacks to use.
    :return: custom callbacks.
    """
    # TODO custom callbacks.
    return []


def get_model(network_args: Dict[str, Any]) -> ProjectModel:
    """Returns the model.
    :param network_args: the network arguments; see DEFAULT_NETWORK_ARGS for
    available arguments.
    :return: the model.
    """
    dataset = ILSVRCDataset(DEFAULT_DATASET_PATH)
    network_args = DEFAULT_NETWORK_ARGS.update(network_args)
    network = MLP(network_args['input_shape'], network_args['num_classes'])
    return ProjectModel(dataset, network)


def train_model(model: ProjectModel, train_args: Dict[str, Any],
                use_wandb: bool = False) -> None:
    """Trains the model.
    :param model: the model to train.
    :param train_args: training arguments; see DEFAULT_TRAIN_ARGS for
    available arguments.
    :param use_wandb: whether to sync the training run to wandb.
    """
    callbacks = []
    if train_args['early_stopping']:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01,
                                       patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)
    if use_wandb:
        callbacks.append(WandbCallback())
        callbacks.extend(get_custom_wandb_callbacks())
    train_args = DEFAULT_TRAIN_ARGS.update(train_args)
    model.network.summary()
    t_start = time()
    _history = model.fit(train_args, callbacks=callbacks)
    t_end = time()
    print('Model training finished in {0:2f}s'.format(t_end - t_start))


def main() -> None:
    """Runs the program."""
    # TODO actually use network_args
    model = get_model({})
    # TODO actually use train_args
    train_model(model, {})


if __name__ == '__main__':
    main()
