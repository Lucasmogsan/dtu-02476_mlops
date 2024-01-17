import torch
from tests import _PROJECT_NAME
from utility.util_functions import set_directories
from train_model import train
from hydra import initialize, compose

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set directories
processed_path, outputs_dir, models_dir, visualization_dir = set_directories()
config_path = "../" + _PROJECT_NAME + "/config/"


# Assert that hydra configurations are loaded correctly
def test_train_config() -> None:
    # config path is relative to this file
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="default_config.yaml")
        hparams = cfg.experiment
        assert hparams == {
            "dataset_path": "data/processed",
            "batch_size": 64,
            "epochs": 3,
            "lr": 1e-3,
            "seed": 123,
            "model_name": "eva02_tiny_patch14_224",
            "classes": [0, 1, 2, 3, 4],
            "test_size": 0.2,
            "val_size": 0.25,
            "n_samples": 500,
        }, "Configurations should match"


# Assert training loss drops after 3 epochs on smaller dataset
def test_train_loss() -> None:
    """Train a model on subset of data found in tests/data"""
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="test_config.yaml")
        train_loss = train(cfg, job_type="unittest")
        assert train_loss[0] > train_loss[-1], "Training loss should decrease after 3 epochs"
