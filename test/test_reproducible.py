"""Tests that training results are reproducible.
Don't add any other tests, because that could mess with the random seeding.
Also, updating packages can mess up the results, but they should still be the
same every time."""

from dataset.ilsvrc_dataset import ILSVRCDataset, DEFAULT_DATASET_PATH
from models.networks.mlp import MLP
from models.project_model import ProjectModel
from training import train_model
from util.util import set_random_seed

SEED = 52017
SEED_HISTORY = "{'loss': [6.760105133056641, 1.0928313732147217, 1.2394728660583496, 0.8879892230033875], 'accuracy': [0.7902321815490723, 0.7870296239852905, 0.8126501441001892, 0.8462769985198975], 'val_loss': [15.447237968444824, 47.05992126464844, 22.719436645507812, 30.461416244506836], 'val_accuracy': [0.0, 0.0, 0.0, 0.0]}"


def test_training_reproducible() -> None:
    """Tests that training results are reproducible."""
    set_random_seed(SEED)
    dataset_args = {'dataset_fraction': 0.001}
    network_args = {'input_shape': (128, 128, 3),
                    'num_classes': 1000}
    train_args = {'epochs': 10, 'batch_size': 32, 'early_stopping': True}
    dataset = ILSVRCDataset(DEFAULT_DATASET_PATH)
    dataset.trim_dataset(dataset_args['dataset_fraction'])
    network = MLP(network_args)
    model = ProjectModel(dataset, network)
    history = train_model.train_model(model, train_args)
    assert str(history.history) == SEED_HISTORY
