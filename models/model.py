"""Model class."""

import os
from typing import Callable, List, Optional
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.callbacks import History
from tensorflow.keras.optimizers import Optimizer, RMSprop

from dataset.dataset import Dataset, TRAIN_KEY, VAL_KEY, TEST_KEY
from dataset.dataset_sequence import DatasetSequence, DEFAULT_BATCH_SIZE

DEFAULT_EPOCHS = 10


class Model:
    """Represents an ML model that can be trained and make predictions."""

    def __init__(self, dataset: Dataset, network: KerasModel) -> None:
        """Instantiates the object.
        :param dataset: the dataset on which to train.
        :param network: the neural network to use.
        """
        self.dataset: Dataset = dataset
        self.network: KerasModel = network
        self.loss: str = 'categorical crossentropy'
        self.optimizer: Optimizer = RMSprop()
        self.metrics: List[str] = ['accuracy']
        self.weights_filename = os.path.join('saved', '{0}_{1}_{2}'.format(
            self.__class__.__name__, self.dataset.__class__.__name__,
            self.network.__class__.__name__
        ))
        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    def fit(self, batch_size: int = DEFAULT_BATCH_SIZE,
            epochs: int = DEFAULT_EPOCHS, augment_val: bool = True,
            callbacks: List[Callback] = None) -> History:
        """Trains the model and returns the history.
        :param batch_size: the number of examples in a batch.
        :param epochs: the number of complete passes over the data.
        :param augment_val: whether to apply the data augmentation
        function to the validation data.
        :param callbacks: a list of keras callbacks to use during
        training.
        :return: the training history.
        """
        callbacks = [] if callbacks is None else callbacks
        self.network.compile(loss=self.loss, optimizer=self.optimizer,
                             metrics=self.metrics)
        x_train_filenames = self.dataset.partition[TRAIN_KEY]
        y_train = self.dataset.get_labels(x_train_filenames)
        x_val_filenames = self.dataset.partition[VAL_KEY]
        y_val = self.dataset.get_labels(x_val_filenames)
        train_sequence = DatasetSequence(
            x_train_filenames, y=y_train, batch_size=batch_size,
            batch_augment_fn=self.batch_augment_fn,
            batch_format_fn=self.batch_format_fn)
        val_sequence = DatasetSequence(
            x_val_filenames, y=y_val, batch_size=batch_size,
            batch_augment_fn=self.batch_augment_fn if augment_val else None,
            batch_format_fn=self.batch_format_fn
        )
        return self.network.fit(
            x=train_sequence,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_sequence,
            use_multiprocessing=False,
            workers=1,
            shuffle=True
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
        sequence = DatasetSequence(x_filenames, y=y, batch_size=batch_size)
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
        sequence = DatasetSequence(x_filenames, y=None, batch_size=batch_size)
        return self.network.predict(sequence)

    def save_weights(self) -> None:
        """Saves the weights to the predefined file."""
        self.network.save_weights(self.weights_filename)

    def load_weights(self) -> None:
        """Loads the weights from the predefined file."""
        self.network.load_weights(self.weights_filename)
