import torch
import os
import wandb
import torch.nn.functional as F
import torch.optim as optim


def set_directories():
    # Directories
    processed_data_path = "data/processed"  # Where processed data is located
    models_dir = "models"  # Where models are saved
    outputs_dir = "outputs"  # Where outputs are saved
    visualization_dir = "reports/figures"  # Where visualizations are saved

    # Create directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(models_dir + "/checkpoints"):
        os.makedirs(models_dir + "/checkpoints")
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    # if not os.path.exists(outputs_dir + "/hydra"):
    #     os.makedirs(outputs_dir + "/hydra")

    return processed_data_path, outputs_dir, models_dir, visualization_dir


def load_data(classes_to_train: list[int], batch_size: int, processed_path: str, job_type: str, seed: int):
    if job_type == "train":
        file_name = "/train_data_"
    elif job_type == "val":
        file_name = "/val_data_"
    elif job_type == "test":
        file_name = "/test_data_"
    elif job_type == "unittest":
        file_name = "/train_data_"
    else:
        raise ValueError("job_type should be one of 'train', 'val', 'test'")

    dataset = torch.load(processed_path + file_name + str(classes_to_train[0]) + ".pt")
    # Iterate over the rest of the classes and concatenate them
    if len(classes_to_train) > 1:
        for _, i in enumerate(classes_to_train):
            if i == classes_to_train[0]:
                continue
            dataset_intermediate = torch.load(processed_path + file_name + str(i) + ".pt")
            dataset = torch.utils.data.ConcatDataset([dataset, dataset_intermediate])
    # Convert to dataloader (to convert to batches and shuffle)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed,
    )

    return dataloader


def log_test_predictions(
    images,
    labels,
    outputs,
    predicted,
    val_table: wandb.Table,
    log_counter: int,
    class_names: list[str],
    NUM_IMAGES_PER_BATCH: int,
):
    """Log predictions for a batch of images to W&B"""
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
        val_table.add_data(
            img_id,
            wandb.Image(img),
            class_names[pred],
            class_names[label],
            *scores,
        )
        _id += 1
        if _id == NUM_IMAGES_PER_BATCH:
            break


def build_optimizer(model, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
        )
    elif optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
    return optimizer
