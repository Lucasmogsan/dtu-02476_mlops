import torch
from torch.nn.functional import softmax
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import sys

# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# LABLES = ["Jasmine", "Ipsala", "Basmati", "Karacadag", "Arborio"]
# load the class label dict from json file
LABLES = json.load(open("models/classes.json"))


# Use albumentations to augment the image (resize and normalize)
transform = A.Compose(
    [
        A.Resize(width=224, height=224),
        A.Normalize(mean=[0.5], std=[0.25], max_pixel_value=255.0),
        ToTensorV2(),
    ],
)


def predict(model_path, input_image_path: str):
    print("Predicting...")

    model = torch.load(model_path)
    # model = timm.create_model("eva02_base_patch14_224", pretrained=True, num_classes=5, in_chans=1).to(device)
    # torch.save(model.cpu(), "models/model_latest.pt")  # save the model to cpu
    #

    # prepare image
    image = cv2.imread(input_image_path)  # (X, X, 3)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # (X, X, 1), converted to grey scale
    image = transform(image=image)["image"]  # (1, 224, 224), resized and normalized
    image = image.unsqueeze(0)  # (1, 1, 224, 224), added a dimension for batch size

    with torch.no_grad():
        # Get the model output
        output = model(image.float().to(device))

        # Get the probabilities and the prediction (max probability)
        probabilities = softmax(output[0], dim=0)
        prediction = probabilities.argmax(dim=0).cpu().item()

        print(LABLES[str(prediction)])
        return LABLES[str(prediction)]


if __name__ == "__main__":
    _, model_path, input_image_path = sys.argv
    predict(model_path, input_image_path)
