"""
Creates a small subset of the training data for unit testing purposes.
Reads raw files from data/raw/Rice_Image_Dataset and saves them to tests/data.
"""
import torch
import os
import cv2
import json
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2


def combine_data(dataset_folder_name: str, data_path: str, n_samples: int):
    """Combine the data into a single file for testing"""

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

    for i, class_type in enumerate(classes):
        print("\nProcessing class: ", class_type)
        classes_dict[i] = class_type
        path = os.path.join(dataset_folder_name, class_type)

        # Load images in the directory
        images = []
        count = 0
        for img in tqdm(os.listdir(path)):
            # Load the image with PIL
            image = cv2.imread(os.path.join(path, img))
            # Image to gray scale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = transform(image=image)["image"]
            # Convert the image to a tensor
            image = image.unsqueeze(0)
            images.append(image)
            count += 1
            if count == n_samples:
                break

        labels_train = torch.tensor([i] * len(images))  # Create labels
        images_train = torch.cat(images, dim=0)
        print("Train data shape: ", images_train.shape)  # torch.Size([n_samples, 1, 224, 224])
        print("Train labels shape: ", labels_train.shape)  # torch.Size([n_samples])

        # Make dataset and save
        train_data = torch.utils.data.TensorDataset(images_train, labels_train)
        torch.save(train_data, processed_path + f"/unittest_data_{i}.pt")

        # Data is not combined into 1 file as the utility function load_data()
        # expects a separate file for each class

    # Save dict classes
    with open(processed_path + "/classes.json", "w") as f:
        json.dump(classes_dict, f)

    print("Data saved!")


if __name__ == "__main__":
    # Set the directory
    dataset_folder_name = "data/raw/Rice_Image_Dataset"
    processed_path = "tests/data"
    n_samples = 5
    combine_data(dataset_folder_name, processed_path, n_samples)
