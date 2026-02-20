import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

# Import model
from model import WaterClassifier

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = WaterClassifier().to(device)
model_path = "../models/water_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    return prob

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py path_to_image")
        sys.exit()

    image_path = sys.argv[1]

    probability = predict(image_path)

    print(f"\nWater Probability: {probability*100:.2f}%")

    if probability > 0.5:
        print("Prediction: WATER (Flood-prone region)")
    else:
        print("Prediction: NON-WATER")