"""Runs a hyperparameter sweep to find the best ML model."""

# Random seeds need to be set up at program launch, before other
# imports, because some libraries use random initialization.
from numpy.random import seed
USE_RANDOM_SEED = True
RANDOM_SEED = 52017
if USE_RANDOM_SEED:
    seed(RANDOM_SEED)


def main() -> None:
    """Runs the program."""
    # TODO


if __name__ == '__main__':
    main()
