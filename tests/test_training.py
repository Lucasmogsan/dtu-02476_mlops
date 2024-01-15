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
            "dataset_path": "~/datasets",
            "batch_size": 64,
            "epochs": 3,
            "lr": 1e-3,
            "x_dim": 784,
            "hidden_dim": 400,
            "latent_dim": 50,
            "seed": 123,
            "model_name": "model_latest1",
            "classes": [0, 1, 2, 3, 4],
        }, "Configurations should match"


# Assert training loss drops across first 2 batches
def test_train_loss() -> None:
    """Train a model on processed data"""
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="default_config.yaml")
        train_loss = train(cfg, job_type="unittest")
        assert train_loss[-1] < train_loss[-2], "Training loss should decrease in the last 2 epochs"
