"""Trains the model."""

from util.util import set_random_seed
USE_RANDOM_SEED = True
if USE_RANDOM_SEED:
    set_random_seed()
# pylint: disable=wrong-import-position
from tensorflow.keras.callbacks import EarlyStopping, Callback
from wandb.keras import WandbCallback
from typing import List
from time import time

from models.model import Model
from dataset.dataset import Dataset

EARLY_STOPPING = True


def get_custom_wandb_callbacks() -> List[Callback]:
    """Returns a list of custom wandb callbacks to use.
    :return: custom callbacks.
    """
    # TODO custom callbacks.
    return []


def get_model() -> Model:
    """Returns the model."""


def train_model(model: Model, dataset: Dataset, epochs: int, batch_size: int,
                augment_val: bool = True, use_wandb: bool = False) -> None:
    """Trains the model.
    :param model: the model to train.
    :param dataset: the dataset on which to train.
    :param epochs: the number of complete passes over the dataset.
    :param batch_size: the number of examples in each batch.
    :param augment_val: whether to use data augmentation on the val
    dataset.
    :param use_wandb: whether to sync the training run to wandb.
    """
    callbacks = []
    if EARLY_STOPPING:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01,
                                       patience=3, verbose=1, mode='auto')
        callbacks.append(early_stopping)
    if use_wandb:
        callbacks.append(WandbCallback())
        callbacks.extend(get_custom_wandb_callbacks())
    model.network.summary()
    t_start = time()
    _history = model.fit(dataset, batch_size=batch_size, epochs=epochs,
                         augment_val=augment_val, callbacks=callbacks)
    t_end = time()
    print('Model training finished in {0:2f}s'.format(t_end - t_start))


def main() -> None:
    """Runs the program."""


if __name__ == '__main__':
    main()
