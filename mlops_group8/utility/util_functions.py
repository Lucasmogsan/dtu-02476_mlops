import torch
import os


def set_directories():
    # Directories
    processed_data_path = "data/processed"  # Where processed data is located
    models_dir = "models"  # Where models are saved
    outputs_dir = "outputs"  # Where outputs are saved
    visualization_dir = "reports/figures"  # Where visualizations are saved

    # Create directories if they don't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    return processed_data_path, outputs_dir, models_dir, visualization_dir


def load_data(classes_to_train, batch_size, processed_path, train=True):
    if train:
        file_name = "/train_data_"
    else:
        file_name = "/test_data_"

    dataset = torch.load(processed_path + file_name + str(0) + ".pt")
    # Iterate over the rest of the classes and concatenate them
    if len(classes_to_train) > 1:
        for _, i in enumerate(classes_to_train):
            if i == 0:
                continue
            dataset_intermediate = torch.load(processed_path + file_name + str(i) + ".pt")
            dataset = torch.utils.data.ConcatDataset([dataset, dataset_intermediate])
    # Convert to dataloader (to convert to batches and shuffle)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader
