"""Tests that training results are reproducible.
Don't add any other tests, because that could mess with the random seeding."""

from dataset.ilsvrc_dataset import ILSVRCDataset, DEFAULT_DATASET_PATH
from models.networks.mlp import MLP
from models.project_model import ProjectModel
from training import train_model
from util.util import set_random_seed

SEED = 52017
SEED_HISTORY = "{'loss': [6.61477518081665, 0.7118954062461853, 0.5890428423881531, 0.5843948125839233, 0.5031907558441162], 'accuracy': [0.7902321815490723, 0.8470776677131653, 0.872698187828064, 0.8598878979682922, 0.8855084180831909], 'val_loss': [15.476541519165039, 12.545025825500488, 16.130390167236328, 24.079486846923828, 38.36538314819336], 'val_accuracy': [0.0, 0.0, 0.0, 0.0, 0.0]}"


def test_training_reproducible() -> None:
    """Tests that training results are reproducible."""
    set_random_seed(SEED)
    dataset_args = {'dataset_fraction': 0.001}
    network_args = {'input_shape': (128, 128, 3),
                    'num_classes': 1000}
    train_args = {'epochs': 10, 'batch_size': 32}
    dataset = ILSVRCDataset(DEFAULT_DATASET_PATH)
    dataset.trim_dataset(dataset_args['dataset_fraction'])
    network = MLP(network_args['input_shape'], network_args['num_classes'])
    model = ProjectModel(dataset, network)
    history = train_model.train_model(model, train_args)
    assert str(history.history) == SEED_HISTORY
