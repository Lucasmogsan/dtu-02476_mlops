import torch
from hydra import initialize, compose
import timm
from tests import _PROJECT_ROOT, _PROJECT_NAME

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set directories
config_path = "../" + _PROJECT_NAME + "/config/"
processed_path = _PROJECT_ROOT + "/data/processed"


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
        }


# Assert training loss drops across first 2 batches
def test_train_batch_loss() -> None:
    """Train a model on processed data"""
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name="default_config.yaml")
        hparams = cfg.experiment
        # dataset_path = hparams["dataset_path"]
        # epochs = hparams["epochs"]
        lr = hparams["lr"]
        batch_size = hparams["batch_size"]
        # latent_dim = hparams["latent_dim"]
        # hidden_dim = hparams["hidden_dim"]
        # x_dim = hparams["x_dim"]
        seed = hparams["seed"]
        # model_name = hparams["model_name"]
        classes_to_train = hparams["classes"]

        # Set seed for reproducibility
        torch.manual_seed(seed)

        # Import model
        model = timm.create_model(
            "eva02_tiny_patch14_224",
            pretrained=False,
            num_classes=len(classes_to_train),
            in_chans=1,
        ).to(device)

        # Import data
        file_name = "/train_data_"
        dataset = torch.load(processed_path + file_name + str(0) + ".pt")
        # Iterate over the rest of the classes and concatenate them
        if len(classes_to_train) > 1:
            for _, i in enumerate(classes_to_train):
                if i == 0:
                    continue
                dataset_intermediate = torch.load(processed_path + file_name + str(i) + ".pt")
                dataset = torch.utils.data.ConcatDataset([dataset, dataset_intermediate])
        # Convert to dataloader (to convert to batches and shuffle)
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Train model hyperparameters
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        num_epochs = 3

        # Training loop
        train_loss = []
        batch_i = 1
        for epoch in range(num_epochs):
            for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                batch_i += 1
            print(f"epoch: {epoch+1} loss: {loss}")
            train_loss.append(loss.item())
        assert train_loss[num_epochs] < train_loss[0], "Training loss should decrease after 3 epochs"
