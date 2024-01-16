import torch
import hydra
import wandb
import os
import torch.nn.functional as F
from json import load
from utility.util_functions import set_directories, load_data
# from torch.profiler import profile, tensorboard_trace_handler, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processed_path, _, models_dir, _ = set_directories()


NUM_BATCHES_TO_LOG = 5  # 79
NUM_IMAGES_PER_BATCH = 2  # 128


def log_test_predictions(images, labels, outputs, predicted, val_table, log_counter):
    # obtain confidence scores for all classes
    scores = F.softmax(outputs.data, dim=1)
    log_scores = scores.cpu().numpy()
    log_images = images.cpu().numpy()
    log_labels = labels.cpu().numpy()
    log_preds = predicted.cpu().numpy()
    # adding ids based on the order of the images
    _id = 0
    for img, label, pred, scores in zip(log_images, log_labels, log_preds, log_scores):
        # add required info to data table:
        # id, image pixels, model's guess, true label, scores for all classes
        img_id = str(_id) + "_" + str(log_counter)
        val_table.add_data(img_id, wandb.Image(img), pred, label, *scores)
        _id += 1
        if _id == NUM_IMAGES_PER_BATCH:
            break


@hydra.main(version_base=None, config_path="config", config_name="default_config.yaml")
def validate(cfg):
    """Validate a trained model."""

    print("### Evaluation setup ###")
    # Read hyperparameters for experiment
    hparams = cfg.experiment
    dataset_path = hparams["dataset_path"]
    # epochs = hparams["epochs"]
    # lr = hparams["lr"]
    batch_size = hparams["batch_size"]
    seed = hparams["seed"]
    model_name = hparams["model_name"]
    classes_to_eval = hparams["classes"]

    class_names = load(open(processed_path + "/classes.json", "r"))
    class_names = list(class_names.values())

    # wandb setup
    wandb_cfg = {
        # "epochs": epochs,
        # "learning_rate": lr,
        "batch_size": batch_size,
        # "latent_dim": latent_dim,
        # "hidden_dim": hidden_dim,
        # "x_dim": x_dim,
        "seed": seed,
    }
    wandb.init(
        project="cm_test",
        entity="mlops_group8",
        config=wandb_cfg,
        job_type="val",
        dir="./outputs",
    )

    torch.manual_seed(seed)

    print("Validating ", model_name)

    # Import model
    model = torch.load(os.path.join(models_dir, model_name + ".pt"))

    print("### Loading data ###")
    val_dataloader = load_data(
        classes_to_eval,
        batch_size,
        dataset_path,
        job_type="val",
    )

    # ✨ W&B: Create a Table to store predictions for each validation step
    columns = ["batch_id", "image", "guess", "truth"]
    for class_name in range(len(class_names)):
        columns.append("score_" + class_name)
    val_table = wandb.Table(columns=columns)

    # Validation loop
    val_labels = []
    val_output = []
    log_counter = 0
    with torch.no_grad():
        # with profile(
        #     activities=[ProfilerActivity.CPU],
        #     record_shapes=False,
        #     on_trace_ready=tensorboard_trace_handler("./log/eva"),
        # ) as prof:
        for i, batch in enumerate(val_dataloader):
            print(f"Iteration: {i+1}/{len(val_dataloader)}")
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            predicted = torch.argmax(output.data, dim=1)
            val_labels.append(labels.cpu())
            val_output.append(output.cpu())

            # Log with wandbb
            if log_counter < NUM_BATCHES_TO_LOG:
                log_test_predictions(
                    images,
                    labels,
                    output,
                    predicted,
                    val_table,
                    log_counter,
                )
                log_counter += 1

                # prof.step()

        val_labels = torch.cat(val_labels, dim=0)
        val_output = torch.cat(val_output, dim=0)

    val_top_preds = val_output.argmax(axis=1)
    val_accuracy = (val_top_preds == val_labels).float().mean().item()
    print("Accuracy: ", val_accuracy)

    wandb.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=val_labels.numpy(),
                preds=val_top_preds.numpy(),
                class_names=class_names,
            ),
        },
    )
    wandb.log({"val_acc": val_accuracy})

    # ✨ W&B: Log predictions table to wandb
    wandb.log({"test_predictions": val_table})

    # Print from the 'prof' object created above:
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))


if __name__ == "__main__":
    validate()
