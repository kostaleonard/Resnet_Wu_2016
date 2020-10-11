"""A LeNet Convolutional Neural Network Keras Model."""

from typing import Dict, Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.backend import image_data_format

from dataset.ilsvrc_dataset import EXPECTED_NUM_CLASSES
from dataset.image_dataset_sequence import DEFAULT_TARGET_SIZE

DEFAULT_LENET_ARGS = {
    'input_shape': DEFAULT_TARGET_SIZE + (3,),
    'num_classes': EXPECTED_NUM_CLASSES,
    'filters_1': 32,
    'kernel_1': (3, 3),
    'pool_1': (2, 2),
    'filters_2': 64,
    'kernel_2': (3, 3),
    'pool_2': (2, 2),
    'dropout': 0.2,
    'dense': 128
}


class LeNet(Sequential):
    """A LeNet Convolutional Neural Network."""

    def __init__(self, lenet_args: Dict[str, Any]) -> None:
        """Creates the object.
        :param lenet_args: the LeNet hyperparameters.
        """
        super().__init__()
        lenet_args = {**DEFAULT_LENET_ARGS, **lenet_args}
        if len(lenet_args['input_shape']) != 3:
            raise ValueError('Expected 3 dimensions (width, height, channels),'
                             'but got.'.format(len(lenet_args['input_shape'])))
        # TODO there may be a need to support channels_first.
        if image_data_format() == 'channels_first':
            raise ValueError('Expected channels last in image tensors.')
        self.add(Conv2D(
            lenet_args['filters_1'],
            kernel_size=lenet_args['kernel_1'],
            activation='relu',
            input_shape=lenet_args['input_shape'],
            padding='valid'
        ))
        self.add(MaxPooling2D(
            pool_size=lenet_args['pool_1'],
            padding='valid'
        ))
        self.add(Conv2D(
            lenet_args['filters_2'],
            kernel_size=lenet_args['kernel_2'],
            activation='relu',
            padding='valid'
        ))
        self.add(MaxPooling2D(
            pool_size=lenet_args['pool_2'],
            padding='valid'
        ))
        self.add(Dropout(lenet_args['dropout']))
        self.add(Flatten())
        self.add(Dense(lenet_args['dense'], activation='relu'))
        self.add(Dense(lenet_args['num_classes'], activation='softmax'))
