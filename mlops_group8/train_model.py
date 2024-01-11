import click
import torch
import timm
import os
import matplotlib.pyplot as plt
import hydra
import wandb


@click.group()
def cli():
    """Command line interface."""
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

visualization_dir = "reports/figures"
models_dir = "models"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)
if not os.path.exists("outputs"):
    os.makedirs("outputs")


### TRAINING ###
@click.command()
@hydra.main(version_base=None, config_path="config", config_name="default_config.yaml")
def train(cfg):
    """Train a model on MNIST."""

    # Hydra config
    hparams = cfg.experiment
    # Read hyperparameters for experiment
    # dataset_path = hparams["dataset_path"]
    epochs = hparams["epochs"]
    lr = hparams["lr"]
    batch_size = hparams["batch_size"]
    latent_dim = hparams["latent_dim"]
    hidden_dim = hparams["hidden_dim"]
    x_dim = hparams["x_dim"]
    seed = hparams["seed"]
    model_name = hparams["model_name"]

    # wandb
    wandb_cfg = {
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "x_dim": x_dim,
        "seed": seed,
    }
    wandb.init(
        project="rice_classification",
        entity="mlops_group8",
        config=wandb_cfg,
        job_type="train",
        dir="./outputs",
    )

    # Set seed (for reproducibility)
    torch.manual_seed(seed)

    print("Training day and night")

    # Import model
    model = timm.create_model("resnet50", pretrained=True, num_classes=5).to(device)

    # Import data
    train_dataset = torch.load("data/processed/train.pt")
    train_data = train_dataset[0]
    train_labels = train_data[1]
    # Convert to dataloader (to convert to batches)
    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )

    # Train model hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = epochs

    # For visualization
    train_loss = []

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        train_loss.append(loss.item())
        acc = 1

        wandb.log({"acc": acc, "loss": loss})

    # Prepare plot
    plt.plot(train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")

    # Save model (and plot)
    # If model_name exists, make new name (add _1, _2, etc.)
    if os.path.exists(models_dir + f"/{model_name}.pt"):
        i = 1
        while os.path.exists(models_dir + f"/{model_name}.pt"):
            i += 1
        torch.save(model, models_dir + f"/{model_name}.pt")  # save model
        plt.savefig(visualization_dir + f"/train_loss_{i}.png")  # save plot
    else:
        torch.save(model, models_dir + f"models/{model_name}.pt")
        plt.savefig(visualization_dir + "/train_loss.png")

    return loss.item()


### EVALUATION ###
@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # Import model
    model = torch.load(os.path.join(models_dir, model_checkpoint))

    # Import data
    test_dataset = torch.load("data/processed/test.pt")
    test_data = test_dataset[0]
    test_labels = test_dataset[1]
    # Convert to dataloader (to convert to batches)
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_data, test_labels),
        batch_size=64,
        shuffle=False,
    )

    test_predictions = []
    test_labels = []

    # Evaluation loop
    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            test_predictions.append(output.argmax(dim=1).cpu())
            test_labels.append(labels.cpu())

        test_predictions = torch.cat(test_predictions, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

    print("Accuracy: ", (test_predictions == test_labels).float().mean().item())


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
