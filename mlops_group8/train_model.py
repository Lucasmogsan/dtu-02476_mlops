import torch
import timm
import hydra
import wandb
import matplotlib.pyplot as plt
from utility.util_functions import set_directories, load_data
from datetime import datetime


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
processed_path, outputs_dir, models_dir, visualization_dir = set_directories()


def train(cfg, job_type="train") -> list:
    """Train a model on processed data"""

    print("### Training setup ###")
    # Read hyperparameters for experiment
    hparams = cfg.experiment
    dataset_path = hparams["dataset_path"]
    epochs = hparams["epochs"]
    lr = hparams["lr"]
    batch_size = hparams["batch_size"]
    seed = hparams["seed"]
    # model_name = hparams["model_name"]
    classes_to_train = hparams["classes"]

    # wandb setup
    wandb_cfg = {
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "seed": seed,
    }
    wandb.init(
        project="rice_classification",
        entity="mlops_group8",
        config=wandb_cfg,
        job_type=job_type,
        dir="./outputs",
    )

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Import model
    model = timm.create_model(
        "eva02_tiny_patch14_224",
        pretrained=False,
        num_classes=len(classes_to_train),
        in_chans=1,
    ).to(device)
    # Train model hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("### Loading data ###")
    train_dataloader = load_data(
        classes_to_train,
        batch_size,
        dataset_path,
        job_type,
    )

    # Training loop
    train_loss = []
    print("### Training model ###")
    for epoch in range(epochs):
        for i, batch in enumerate(train_dataloader):
            # print iteration of total iterations for this epoch
            print(
                f"Epoch: {epoch+1}/{epochs}, Iteration: {i+1}/{len(train_dataloader)}",
            )
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
    torch.save(model, models_dir + "/model_latest.pt")  # save as latest model
    date_time = datetime.now().strftime("%Y%m%d_%H%M")
    torch.save(model, models_dir + "/checkpoints/model_" + date_time + ".pt")

    print("### Finished training ###")

    return train_loss


### TRAINING ###
@hydra.main(version_base=None, config_path="config", config_name="default_config.yaml")
def main(cfg):
    """Train a model on processed data"""
    _ = train(cfg)


if __name__ == "__main__":
    main()
