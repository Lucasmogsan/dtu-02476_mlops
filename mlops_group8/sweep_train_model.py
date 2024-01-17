import torch
import timm
import wandb
import yaml
from json import load
from utility.util_functions import set_directories, load_data, log_test_predictions, build_optimizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories
processed_path, outputs_dir, models_dir, visualization_dir = set_directories()

# ✨ W&B: for table logging
NUM_BATCHES_TO_LOG = 5
NUM_IMAGES_PER_BATCH = 5


def train(config=None):
    """Train a model on processed data"""

    with wandb.init(
        config=config,
        dir="./outputs",
    ):
        config = wandb.config

        print("### Training setup ###")

        # Create a list of class names
        class_names = load(open(processed_path + "/classes.json", "r"))
        class_names = list(class_names.values())  # get the names only
        class_names = [class_names[i] for i in config.classes]

        # ✨ W&B: Create a Table to store predictions for each validation step
        columns = ["batch_id", "image", "guess", "truth"]
        for class_name in class_names:
            columns.append("score_" + class_name)
        train_table = wandb.Table(columns=columns)

        # Set seed for reproducibility
        torch.manual_seed(config.seed)

        # Import model
        model = timm.create_model(
            config.model_name,
            pretrained=False,
            num_classes=len(config.classes),
            in_chans=1,
        ).to(device)
        print("Training ", config.model_name)

        # Train model hyperparameters
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = build_optimizer(model, config.optimizer, config.lr)

        # Load data
        print("### Loading data ###")
        train_dataloader = load_data(
            config.classes,
            config.batch_size,
            config.dataset_path,
            job_type="train",
        )

        # Training loop
        train_loss = []
        log_counter = 0
        print("### Training model ###")
        for epoch in range(config.epochs):
            epoch_acc = []
            for i, batch in enumerate(train_dataloader):
                print(
                    f"Epoch: {epoch+1}/{config.epochs}, \
                    Iteration: {i+1}/{len(train_dataloader)}",
                )
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
                epoch_acc.append((batch_top_preds == batch_labels))

                # Log with wandb
                if log_counter < NUM_BATCHES_TO_LOG:
                    log_test_predictions(
                        images,
                        labels,
                        output,
                        batch_top_preds,
                        train_table,
                        log_counter,
                        class_names,
                        NUM_IMAGES_PER_BATCH,
                    )
                    log_counter += 1

            # Appending after each epoch uses too much memory, freezes my computer
            train_loss.append(loss.item())
            epoch_acc = torch.cat(epoch_acc, dim=0).numpy().mean()
            print(f"Epoch acc: {epoch_acc}")
            wandb.log({"acc": epoch_acc, "loss": loss})
        wandb.log({"train_predictions": train_table})

        print("### Finished training ###")


def main():
    """Train a model with hyperparam sweep on processed data"""

    with open("mlops_group8/config/sweep.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)
    print(sweep_config)

    sweep_id = wandb.sweep(sweep=sweep_config, project="my-first-sweep")
    wandb.agent(sweep_id, train, count=2)


if __name__ == "__main__":
    main()
