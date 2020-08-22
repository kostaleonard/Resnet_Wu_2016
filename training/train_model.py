"""Trains the model."""

# Random seeds need to be set up at program launch, before other
# imports, because some libraries use random initialization.
from numpy.random import seed
USE_RANDOM_SEED = True
RANDOM_SEED = 52017
if USE_RANDOM_SEED:
    seed(RANDOM_SEED)
from models.model import Model


def get_model() -> Model:
    """Returns the model."""


def train_model() -> None:
    """Trains the model."""


def main() -> None:
    """Runs the program."""


if __name__ == '__main__':
    main()
