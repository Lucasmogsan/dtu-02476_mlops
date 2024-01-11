import os
from zipfile import ZipFile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
import requests
from tqdm import tqdm
import torch
import cv2
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2




# Set the dataset details
dataset_name = "muratkokludataset/rice-image-dataset"
download_path = "data/raw"
extract_path = "data/raw"

transform = A.Compose([
    A.Resize(width=224, height=224),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2()
])

def get_data():
    # Authenticate with Kaggle API
    api = KaggleApi()
    api.authenticate()

    Path(download_path).mkdir(parents=True, exist_ok=True)
    Path(extract_path).mkdir(parents=True, exist_ok=True)

    # Download the dataset
    print(f"Downloading dataset '{dataset_name}' to '{download_path}'...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=False)

    # Extract the dataset
    print(f"Extracting dataset '{dataset_name}' to '{extract_path}'...")
    zip_file = [file for file in os.listdir(download_path) if file.endswith('.zip')][0]
    with ZipFile(os.path.join(download_path, zip_file), 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    extracted_folders = [name for name in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, name))]
    if extracted_folders:
        dataset_folder_name = extracted_folders[0]  # Assuming there's only one folder in the extraction path
        dataset_folder_path = os.path.join(extract_path, dataset_folder_name)
        print(f"The dataset was extracted to: {dataset_folder_path}")
        return dataset_folder_path
    else:
        print("No folder found in the extraction path.")
        exit(1)
        
    
def process_data(dataset_folder_name):
    #clasess = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    clasess = [f.name for f in os.scandir(dataset_folder_name) if f.is_dir()]

    num_classes = len(clasess)

    print("Number of classes:", num_classes)
    print("Class names:", clasess)
    
    # load images from each class folder
    classes_dict = {}
    data_images_train, data_images_test = [], []
    data_labels_train, data_labels_test = [], []
    for i,class_type in enumerate(clasess):

        print("Processing class: ", class_type)
        classes_dict[i] = class_type
        path = os.path.join(dataset_folder_name, class_type)
        # example of image path data/raw/Rice_Image_Dataset/Arborio/Arborio (1).jpg
        # load all images in the directory
        images = []
        count = 0
        for img in tqdm(os.listdir(path)):
            # Load the image with PIL
            image = cv2.imread(os.path.join(path, img))
            # image to gray scale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            image = transform(image=image)["image"]
            # Convert the image to a tensor
            image = image.unsqueeze(0)
            images.append(image)
            # if count == 10:
            #     break
            # count += 1
            
        
        label = torch.tensor([i]*len(images))
        images_data = torch.cat(images, dim=0)
        print(images_data.shape)
        images_train, images_test, labels_train, labels_test = train_test_split(images_data, label, test_size=0.2, random_state=42)
        #print(images_train.shape)
        
        print("Train data shape: ", images_train.shape)
        print("Test data shape: ", images_test.shape)
        print("Train labels shape: ", labels_train.shape)
        print("Test labels shape: ", labels_test.shape)
                
        train_data = torch.utils.data.TensorDataset(images_train, labels_train)
        test_data = torch.utils.data.TensorDataset(images_test, labels_test)
        
        torch.save(train_data, f"data/processed/train_data_{i}.pt")
        torch.save(test_data, f"data/processed/test_data_{i}.pt")
        
    # save dict classes
    
    with open('data/processed/classes.json', 'w') as f:
        json.dump(classes_dict, f)
    
    print("Data saved!")
    

if __name__ == '__main__':
    # Get the data and process it
    #get_data()
    path = 'data/raw/Rice_Image_Dataset'
    process_data(path)  




