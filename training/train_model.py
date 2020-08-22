"""Trains the model."""

from util.util import set_random_seed
USE_RANDOM_SEED = True
if USE_RANDOM_SEED:
    set_random_seed()
# pylint: disable=wrong-import-position
from models.model import Model


def get_model() -> Model:
    """Returns the model."""


def train_model() -> None:
    """Trains the model."""


def main() -> None:
    """Runs the program."""


if __name__ == '__main__':
    main()
