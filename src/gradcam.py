import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from model import WaterClassifier
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


model = WaterClassifier().to(device)
model.load_state_dict(torch.load("../models/water_model.pth", map_location=device))
model.eval()

# Target layer (last convolution layer of ResNet18)
cam_extractor = GradCAM(model, target_layer="model.layer4")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def generate_gradcam(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()

    class_idx = 0

    # Generate CAM
    activation_map = cam_extractor(class_idx, output)

    # activation_map is a list, take first element
    heatmap = activation_map[0].squeeze().cpu().numpy()

    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Resize to original image size
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))

    # Convert to 8-bit
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Convert to color heatmap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert PIL image to OpenCV format
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Overlay
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)

    # Convert back to RGB for matplotlib
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6,6))
    plt.imshow(overlay)
    plt.title(f"Water Probability: {prob*100:.2f}%")
    plt.axis("off")
    plt.show()