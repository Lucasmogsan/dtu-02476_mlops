import timm 
import torch
from torch.nn.functional import softmax
import cv2
import click
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Create a click group
@click.group()
def cli():
    """Command line interface."""
    pass


# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LABLES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

# Use albumentations to augment the image (resize and normalize)
transform = A.Compose([
    A.Resize(width=224, height=224),
    A.Normalize(),
    ToTensorV2()
])


### Predict ###
@click.command()
@click.argument("input_path")
def predict(input_path: str):
    print("Predicting...")
    
    model = torch.load("models/model_latest.pt")
    #model = timm.create_model("resnet50", pretrained=True, num_classes=5)

    # save the model to cpu
    #torch.save(model.cpu(), "models/model_latest.pt")

    with torch.no_grad():
        image = cv2.imread(input_path)  # (X, X, 3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (X, X, 3), now in rgb instead of bgr

        image = transform(image=image)["image"] # (3, 224, 224), resized and normalized
        image = image.unsqueeze(0)  # (1, 3, 224, 224), added a dimension for batch size

        # Get the model output
        output = model(image.float().to(device))
        print(output)

        # Get the probabilities
        probabilities = softmax(output[0], dim=0)
        print(probabilities)

        prediction = probabilities.argmax(dim=0).cpu().item()
        print(prediction)

        print(LABLES[prediction])


# Add commands to the group
cli.add_command(predict)

if __name__ == "__main__":
    cli()

# RUN: python mlops_group8/predict_model.py predict data/raw/Arborio/Arborio\ \(1\).jpg 