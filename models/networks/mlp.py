"""An MLP Keras Model."""

from typing import Dict, Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

from dataset.ilsvrc_dataset import EXPECTED_NUM_CLASSES
from dataset.image_dataset_sequence import DEFAULT_TARGET_SIZE

DEFAULT_MLP_ARGS = {
    'input_shape': DEFAULT_TARGET_SIZE + (3,),
    'num_classes': EXPECTED_NUM_CLASSES,
    'layer_size': 128,
    'dropout_rate': 0.2,
    'num_layers': 3
}


class MLP(Sequential):
    """A Multi-layer Perceptron."""

    def __init__(self, mlp_args: Dict[str, Any]) -> None:
        """Creates the object.
        :param mlp_args: the MLP hyperparameters.
        """
        super().__init__()
        mlp_args = {**DEFAULT_MLP_ARGS, **mlp_args}
        self.add(Flatten(input_shape=mlp_args['input_shape']))
        for _ in range(mlp_args['num_layers']):
            self.add(Dense(mlp_args['layer_size'], activation='relu'))
            self.add(Dropout(mlp_args['dropout_rate']))
        self.add(Dense(mlp_args['num_classes'], activation='softmax'))
