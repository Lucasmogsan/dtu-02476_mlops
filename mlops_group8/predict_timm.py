import timm 
import torch
import os
import cv2
import numpy as np
import click



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths:
current_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.join(current_dir, "models")

package_dir = os.path.join(current_dir, "../")
raw_data_dir = os.path.join(package_dir, "data/raw/corruptmnist")
processed_data_dir = os.path.join(package_dir, "data/processed")
visualization_dir = os.path.join(package_dir, "reports/figures")

# Configs:
LABELS = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]



@click.command()
@click.argument("model_name") # help="name of model to evaluate - remember to add .pt and should be loated in models folder"
@click.argument("input_images") # help="path from package root to input images to predict on - remember to add .pt"
def predict(model_name: torch.nn.Module, input_images):
    print("Predicting...")

    ### Import model ###
    model_path =  os.path.join(os.path.dirname(__file__), "../../models/model_latest.pt")
    model = timm.create_model("resnet50", pretrained=True, checkpoint_path=model_path, num_classes=5 )

    ### Import data ###
    input_path = os.path.join(package_dir, input_images)
    if input_path.endswith(".pt"):  # If input path ends with .pt, it is a tensor
        input_data = torch.load(input_path)
    elif input_path.endswith(".npy"):  # If input path ends with .npy, it is a numpy array
        input_data = torch.from_numpy(np.load(input_path)).unsqueeze(1)
    # TODO: What to do with other file types? jpg, png, etc.

    ### Predict ###
    predictions = []

    with torch.no_grad():
        for image in input_data:
            image = image.to(device).unsqueeze(1)

            output = model(image.float().unsqueeze(0))

            predictions.append(output.argmax(dim=1).cpu())

    predictions = torch.cat(predictions, dim=0)

    print("Predictions: ", predictions)
    return predictions


# RUN: python mlops_group8/predict_model.py predict data/raw/Arborio/Arborio\ \(1\).jpg 