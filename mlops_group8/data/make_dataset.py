import os
import torch
import cv2
import json
import hydra
import albumentations as A
from zipfile import ZipFile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

_SRC_PATH = os.path.dirname(os.path.dirname(__file__))


@hydra.main(version_base=None, config_path=_SRC_PATH + "/config", config_name="default_config.yaml")
def hydra_cfg(cfg) -> dict:
    return cfg.experiment
    # BUG: somehow returns a None type


def download_data(dataset_name: str, download_path: str, extract_path: str) -> None:
    """Download the dataset from Kaggle"""
    # Authenticate with Kaggle API
    api = KaggleApi()
    api.authenticate()

    Path(download_path).mkdir(parents=True, exist_ok=True)
    Path(extract_path).mkdir(parents=True, exist_ok=True)

    # Download the dataset
    print(f"Downloading dataset '{dataset_name}' to '{download_path}'...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=False)


def extract_data(download_path: str, extract_path: str, dataset_name: str) -> None:
    """Extract the dataset"""
    print(f"Extracting dataset '{dataset_name}' to '{extract_path}'...")
    zip_file = [
        file
        for file in os.listdir(
            download_path,
        )
        if file.endswith(".zip")
    ][0]
    with ZipFile(os.path.join(download_path, zip_file), "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Delete zip file
    print("Deleting zip file: ", download_path + "/" + zip_file)
    os.remove(download_path + "/" + zip_file)

    extracted_folders = [
        name
        for name in os.listdir(
            extract_path,
        )
        if os.path.isdir(os.path.join(extract_path, name))
    ]
    if extracted_folders:
        # Assuming there's only one folder in the extraction path
        dataset_folder_name = extracted_folders[0]
        dataset_folder_path = os.path.join(extract_path, dataset_folder_name)
        print(f"The dataset was extracted to: {dataset_folder_path}")
    else:
        print("No folder found in the extraction path.")
        exit(1)


def process_data(
    dataset_folder_name: str,
    processed_path: str,
    n_samples=100,
    test_size=0.2,
    val_size=0.25,
) -> None:
    """
    Create a dataset (images, labels) containing n_samples for each class,
    and split into train, validation and test sets.
    """

    # classes = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    classes = [f.name for f in os.scandir(dataset_folder_name) if f.is_dir()]
    n_classes = len(classes)

    print("Number of classes:", n_classes)
    print("Class names:", classes)

    # Load images from each class folder
    classes_dict = {}

    transform = A.Compose(
        [
            A.Resize(width=224, height=224),
            A.Normalize(mean=[0.5], std=[0.25]),
            ToTensorV2(),
        ],
    )

    for i, class_name in enumerate(classes):
        print("\nProcessing class: ", class_name)
        classes_dict[i] = class_name
        path = os.path.join(dataset_folder_name, class_name)
        # Example of image path data/raw/Rice_Image_Dataset/Arborio/Arborio (1).jpg
        # Load images in the directory up to n_samples
        images = []
        count = 0
        for img in tqdm(os.listdir(path)):
            # Load the image with PIL
            image = cv2.imread(os.path.join(path, img))
            # Convert to gray scale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = transform(image=image)["image"]
            # Convert to a tensor
            image = image.unsqueeze(0)
            images.append(image)
            count += 1
            if count == n_samples:
                break

        label = torch.tensor([i] * len(images))
        images_data = torch.cat(images, dim=0)

        images_train, images_test, labels_train, labels_test = train_test_split(
            images_data,
            label,
            test_size=test_size,
            random_state=42,
        )
        images_train, images_val, labels_train, labels_val = train_test_split(
            images_train,
            labels_train,
            test_size=val_size,
            random_state=42,
        )

        print("Train data shape: ", images_train.shape)  # [300, 1, 224, 224]
        print("Train labels shape: ", labels_train.shape)  # [300]
        print("Validation data shape: ", images_val.shape)  # [100, 1, 224, 224]
        print("Validation labels shape: ", labels_val.shape)  # [100]
        print("Test data shape: ", images_test.shape)  # [100, 1, 224, 224]
        print("Test labels shape: ", labels_test.shape)  # [100]

        train_data = torch.utils.data.TensorDataset(images_train, labels_train)
        val_data = torch.utils.data.TensorDataset(images_val, labels_val)
        test_data = torch.utils.data.TensorDataset(images_test, labels_test)

        torch.save(train_data, processed_path + f"/train_data_{i}.pt")
        torch.save(val_data, processed_path + f"/val_data_{i}.pt")
        torch.save(test_data, processed_path + f"/test_data_{i}.pt")

    # Save dict classes
    with open(processed_path + "/classes.json", "w") as f:
        json.dump(classes_dict, f)

    print("Data saved!")


if __name__ == "__main__":
    # Set the dataset details
    dataset_name = "muratkokludataset/rice-image-dataset"
    download_path = "data/raw"
    extract_path = "data/raw"
    processed_path = "data/processed"

    # Create the directories if they don't exist
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    cfg = hydra_cfg()
    test_size = cfg["test_size"]
    hparams = cfg.experiment
    test_size = hparams["test_size"]
    val_size = hparams["val_size"]
    n_samples = hparams["n_samples"]

    # Get the data and process it
    path = extract_path + "/Rice_Image_Dataset"
    if not os.path.exists(path):
        download_data(download_path, extract_path, dataset_name)
        extract_data(download_path, extract_path, dataset_name)
    process_data(path, processed_path, n_samples, test_size, val_size)
