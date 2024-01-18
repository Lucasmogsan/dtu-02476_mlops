import torch
import timm
from tests import _PROJECT_NAME
from utility.util_functions import set_directories, load_data
from train_model import train_epoch
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
        cfg = compose(config_name="test_config.yaml")
        hparams = cfg.experiment
        assert hparams == {
            "dataset_path": "tests/data",
            "batch_size": 64,
            "epochs": 3,
            "lr": 1e-4,
            "seed": 123,
            "model_name": "eva02_tiny_patch14_224",
            "classes": [0, 1, 2, 3, 4],
            "test_size": 0.2,
            "val_size": 0.25,
            "n_samples": 500,
            "optimizer": "adam",
        }, "Configurations should match"


# Assert training loss drops after 3 epochs on smaller dataset
def test_train_loss() -> None:
    """Train a model on subset of data found in tests/data"""
    losses = []
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="test_config.yaml")
        hparams = cfg.experiment
        dataset_path = hparams["dataset_path"]
        epochs = hparams["epochs"]
        lr = hparams["lr"]
        batch_size = hparams["batch_size"]
        seed = hparams["seed"]
        model_name = hparams["model_name"]
        classes_to_train = hparams["classes"]
        optimizer_name = hparams["optimizer"]

        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=len(classes_to_train),
            in_chans=1,
        ).to(device)

        unit_dataloader = load_data(
            classes_to_train,
            batch_size,
            dataset_path,
            "unittest",
            seed,
        )
        for _ in range(epochs):
            _, train_loss = train_epoch(model, unit_dataloader, optimizer_name, lr)
            losses.append(train_loss)
        assert losses[0] > losses[-1], "Training loss should decrease after 3 epochs"
