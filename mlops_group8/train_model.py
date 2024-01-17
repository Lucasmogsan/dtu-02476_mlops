import torch
import timm
import hydra
import wandb
from datetime import datetime
from json import load
from utility.util_functions import set_directories, load_data, log_test_predictions, build_optimizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
processed_path, outputs_dir, models_dir, visualization_dir = set_directories()

# ✨ W&B: for table logging
NUM_BATCHES_TO_LOG = 5
NUM_IMAGES_PER_BATCH = 5


def train_epoch(model, train_dataloader, optimizer_name, lr):
    # Train model hyperparameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_name, lr)

    print("### Training model ###")
    train_acc = []
    for i, batch in enumerate(train_dataloader):
        print(f"Iteration: {i+1}/{len(train_dataloader)}")
        optimizer.zero_grad()
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # Save labels and predictions
        batch_labels = torch.cat([labels], dim=0)
        batch_output = torch.cat([output], dim=0)
        batch_top_preds = batch_output.argmax(axis=1)
        train_acc.append((batch_top_preds == batch_labels))

    # Appending after each epoch uses too much memory -> computer freezes
    train_acc = torch.cat(train_acc, dim=0).numpy().mean()
    print(f"train_acc: {train_acc}")

    return train_acc, loss


def validate_epoch(model, val_dataloader, val_table, class_names):
    # Validation loop after each epoch
    print("### Validating model ###")
    val_labels = []
    val_output = []
    log_counter = 0
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            print(f"Iteration: {i+1}/{len(val_dataloader)}")
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            predicted = torch.argmax(output.data, dim=1)
            val_labels.append(labels.cpu())
            val_output.append(output.cpu())

            # Log with wandbb=
            if log_counter < NUM_BATCHES_TO_LOG:
                log_test_predictions(
                    images,
                    labels,
                    output,
                    predicted,
                    val_table,
                    log_counter,
                    class_names,
                    NUM_IMAGES_PER_BATCH,
                )
                log_counter += 1

        val_labels = torch.cat(val_labels, dim=0)
        val_output = torch.cat(val_output, dim=0)

    val_top_preds = val_output.argmax(axis=1)
    val_acc = (val_top_preds == val_labels).float().mean().item()
    print("val_acc: ", val_acc)

    print("### Finished validation ###")
    return val_acc


def test_model(model, test_dataloader, class_names):
    """Test model on test set"""
    print("### Testing model ###")
    test_labels = []
    test_output = []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            print(f"Iteration: {i+1}/{len(test_dataloader)}")
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            test_labels.append(labels.cpu())
            test_output.append(output.cpu())

        test_labels = torch.cat(test_labels, dim=0)
        test_output = torch.cat(test_output, dim=0)

    test_top_preds = test_output.argmax(axis=1)
    test_acc = (test_top_preds == test_labels).float().mean().item()
    print("test_acc: ", test_acc)

    # ✨ W&B: Logging for test step
    wandb.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_labels.numpy(),
                preds=test_top_preds.numpy(),
                class_names=class_names,
            ),
        },
    )
    print("### Finished testing ###")
    return test_acc


def save_model(model, date_time):
    print("### Saving model ###")
    # Save as latest model
    torch.save(model, models_dir + "/model_latest.pt")
    # Save as model_YYYYmm_HHMM.pt
    torch.save(model, models_dir + "/checkpoints/model_" + date_time + ".pt")


@hydra.main(version_base=None, config_path="config", config_name="default_config.yaml")
def main(cfg):
    """Train a model on processed data"""
    date_time = datetime.now().strftime("%Y%m%d_%H%M")

    print("### Training setup ###")
    # Read hyperparameters for experiment
    hparams = cfg.experiment
    dataset_path = hparams["dataset_path"]
    epochs = hparams["epochs"]
    lr = hparams["lr"]
    batch_size = hparams["batch_size"]
    seed = hparams["seed"]
    model_name = hparams["model_name"]
    classes_to_train = hparams["classes"]
    optimizer_name = hparams["optimizer"]

    # Create a list of class names
    class_names = load(open(processed_path + "/classes.json", "r"))
    class_names = list(class_names.values())  # get the names only
    class_names = [class_names[i] for i in classes_to_train]

    # ✨ W&B: setup
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
        job_type="train",
        name="train_" + date_time,
        dir="./outputs",
    )

    # ✨ W&B: Create a Table to store predictions for each validation step
    columns = ["batch_id", "image", "guess", "truth"]
    for class_name in class_names:
        columns.append("score_" + class_name)
    val_table = wandb.Table(columns=columns)

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Import model
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(classes_to_train),
        in_chans=1,
    ).to(device)

    # Load data
    print("### Loading data ###")
    train_dataloader = load_data(
        classes_to_train,
        batch_size,
        dataset_path,
        "train",
        seed,
    )
    val_dataloader = load_data(
        classes_to_train,
        batch_size,
        dataset_path,
        "val",
        seed,
    )
    test_dataloader = load_data(
        classes_to_train,
        batch_size,
        dataset_path,
        "test",
        seed,
    )

    # Training loop
    train_losses = []  # needed for test_training.py
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")

        train_acc, train_loss = train_epoch(model, train_dataloader, optimizer_name, lr)
        wandb.log({"train_acc": train_acc, "train_loss": train_loss})
        train_losses.append(train_loss)

        val_acc = validate_epoch(model, val_dataloader, val_table, class_names)
        wandb.log({"val_acc": val_acc})

    wandb.log({"val_predictions": val_table})

    # Test loop
    test_acc = test_model(model, test_dataloader, class_names)
    wandb.log({"test_acc": test_acc})

    # Save model
    save_model(model, date_time)

    print("### Finished training ###")

    return train_losses


if __name__ == "__main__":
    main()
