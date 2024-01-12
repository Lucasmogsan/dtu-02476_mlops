import torch
import timm
import os
import matplotlib.pyplot as plt
import hydra
import wandb
from utility.util_functions import set_directories, load_data


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
processed_path, outputs_dir, models_dir, visualization_dir = set_directories()


### TRAINING ###
@hydra.main(version_base=None, config_path="config", config_name="default_config.yaml")
def main(cfg):
    """Train a model on processed data"""

    print("### Training setup ###")
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
    classes_to_train = hparams["classes_to_train"]

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

    print("### Loading data ###")

    # Import model
    model = timm.create_model(
        "eva02_tiny_patch14_224",
        pretrained=False,
        num_classes=len(classes_to_train),
        in_chans=1,
    ).to(device)

    # Import data
    train_dataloader = load_data(classes_to_train, batch_size, processed_path, train=True)

    # Train model hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = epochs

    # For visualization
    train_loss = []

    print("### Training model ###")
    # Training loop
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_dataloader):
            # print iteration of total iterations for this epoch
            print(f"Epoch: {epoch+1}/{num_epochs}, Iteration: {i+1}/{len(train_dataloader)}")
            optimizer.zero_grad()

            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            loss = criterion(output, labels)

            loss.backward()

            optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        train_loss.append(loss.item())
        acc = 1

        wandb.log({"acc": acc, "loss": loss})

    # Prepare plot
    print("### Make visualizations ###")
    plt.plot(train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")

    print("### Saving model and plot ###")

    # Save model (and plot)
    # If model_name exists, make new name (add _1, _2, etc.)
    if os.path.exists(models_dir + f"/{model_name}.pt"):
        i = 1
        while os.path.exists(models_dir + f"/{model_name}{i}.pt"):
            i += 1
        torch.save(model, models_dir + f"/{model_name}{i}.pt")  # save model
        plt.savefig(visualization_dir + f"/train_loss{i}.png")  # save plot
    else:
        torch.save(model, models_dir + f"/{model_name}.pt")
        plt.savefig(visualization_dir + "/train_loss.png")

    print("### Finished ###")


if __name__ == "__main__":
    main()
