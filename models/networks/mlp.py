"""An MLP Keras Model."""

from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout


class MLP(Sequential):
    """A Multi-layer Perceptron."""

    def __init__(self,
                 input_shape: Tuple[int, ...],
                 num_classes: int,
                 layer_size: int = 128,
                 dropout_rate: float = 0.2,
                 num_layers: int = 3):
        """Creates the object."""
        super().__init__()
        self.add(Flatten(input_shape=input_shape))
        for _ in range(num_layers):
            self.add(Dense(layer_size, activation='relu'))
            self.add(Dropout(dropout_rate))
        self.add(Dense(num_classes, activation='softmax'))
