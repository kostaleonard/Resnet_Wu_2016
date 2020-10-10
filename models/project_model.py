"""Model class."""

import os
from typing import Callable, List, Optional, Dict, Any
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Optimizer, RMSprop

from dataset.dataset import Dataset, TRAIN_KEY, VAL_KEY, TEST_KEY
from dataset.image_dataset_sequence import ImageDatasetSequence, \
    DEFAULT_BATCH_SIZE

DEFAULT_TRAIN_ARGS = {
    'batch_size': DEFAULT_BATCH_SIZE,
    'epochs': 10,
    'augment_val': True,
    'early_stopping': True,
    'overfit_single_batch': False,
    'shuffle_on_epoch_end': True
}


class ProjectModel:
    """Represents an ML model that can be trained and make predictions."""

    def __init__(self, dataset: Dataset, network: KerasModel) -> None:
        """Instantiates the object.
        :param dataset: the dataset on which to train.
        :param network: the neural network to use.
        """
        self.dataset: Dataset = dataset
        self.network: KerasModel = network
        self.loss: str = 'categorical_crossentropy'
        self.optimizer: Optimizer = RMSprop()
        self.metrics: List[str] = ['accuracy']
        self.weights_filename = os.path.join('saved', '{0}_{1}_{2}'.format(
            self.__class__.__name__, self.dataset.__class__.__name__,
            self.network.__class__.__name__
        ))
        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    def fit(self, train_args: Dict[str, Any],
            callbacks: List[Callback] = None) -> History:
        """Trains the model and returns the history.
        :param train_args: the training arguments.
        :param callbacks: a list of keras callbacks to use during
        training.
        :return: the training history.
        """
        train_args = {**DEFAULT_TRAIN_ARGS, **train_args}
        callbacks = [] if callbacks is None else callbacks
        self.network.compile(loss=self.loss, optimizer=self.optimizer,
                             metrics=self.metrics)
        x_train_filenames = self.dataset.partition[TRAIN_KEY]
        y_train = self.dataset.get_labels(x_train_filenames, True,
                                          self.network.output_shape[1])
        x_val_filenames = self.dataset.partition[VAL_KEY]
        y_val = self.dataset.get_labels(x_val_filenames, True,
                                        self.network.output_shape[1])
        train_sequence = ImageDatasetSequence(
            x_train_filenames, y=y_train, batch_size=train_args['batch_size'],
            image_target_size=self.network.input_shape[1:3],
            batch_augment_fn=self.batch_augment_fn,
            batch_format_fn=self.batch_format_fn,
            overfit_single_batch=train_args['overfit_single_batch'],
            shuffle_on_epoch_end=train_args['shuffle_on_epoch_end']
        )
        val_sequence = ImageDatasetSequence(
            x_val_filenames, y=y_val, batch_size=train_args['batch_size'],
            image_target_size=self.network.input_shape[1:3],
            batch_augment_fn=self.batch_augment_fn if train_args['augment_val']
            else None,
            batch_format_fn=self.batch_format_fn,
            shuffle_on_epoch_end=train_args['shuffle_on_epoch_end']
        )
        return self.network.fit(
            x=train_sequence,
            epochs=train_args['epochs'],
            callbacks=callbacks,
            validation_data=val_sequence,
            use_multiprocessing=False,
            workers=1
        )

    def evaluate(self, x_filenames: np.ndarray, y: np.ndarray,
                 batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """Evaluates the model on the given dataset.
        :param x_filenames: an np.ndarray of strs where each str is an
        image filename.
        :param y: an np.ndarray of ints where the ith int is the label
        of the ith image in x_filenames.
        :param batch_size: the number of examples in each batch.
        :return: the mean number of correct predictions.
        """
        # TODO y should be categorical.
        sequence = ImageDatasetSequence(
            x_filenames, y=y, batch_size=batch_size,
            image_target_size=self.network.input_shape()[:2])
        preds = self.network.predict(sequence)
        return np.mean(np.argmax(preds, axis=-1) == np.argmax(y, axis=-1))

    def predict(self, x_filenames: np.ndarray,
                batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """Makes a prediction on the given examples.
        :param x_filenames: an np.ndarray of strs where each str is an
        image filename.
        :param batch_size: the number of examples in each batch.
        :return: an np.ndarray of predicted classes.
        """
        sequence = ImageDatasetSequence(
            x_filenames, y=None, batch_size=batch_size,
            image_target_size=self.network.input_shape()[:2])
        return self.network.predict(sequence)

    def predict_on_test(self,
                        batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """Makes a prediction on the test dataset.
        :param batch_size: the number of examples in each batch.
        :return: an np.ndarray of predicted classes.
        """
        return self.predict(self.dataset.partition[TEST_KEY],
                            batch_size=batch_size)

    def save_weights(self) -> None:
        """Saves the weights to the predefined file."""
        self.network.save_weights(self.weights_filename)

    def load_weights(self) -> None:
        """Loads the weights from the predefined file."""
        self.network.load_weights(self.weights_filename)
