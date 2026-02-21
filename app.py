import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
import cv2
from torchcam.methods import GradCAM

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.model import WaterClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model path (local)
MODEL_PATH = "models/water_model.pth"

model = WaterClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

cam_extractor = GradCAM(model, target_layer="model.layer4")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("🌊 Satellite Flood Risk + GradCAM (Local)")

uploaded_files = st.file_uploader(
    "Upload Satellite Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:

        image = Image.open(uploaded_file).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        img_tensor.requires_grad = True
        output = model(img_tensor)
        probability = torch.sigmoid(output).item()

        st.markdown("---")
        st.subheader(uploaded_file.name)

        st.write(f"Water Probability: {probability*100:.2f}%")
        st.progress(probability)

        if st.button(f"Show GradCAM for {uploaded_file.name}"):

            activation_map = cam_extractor(0, output)
            heatmap = activation_map[0].squeeze().cpu().numpy()

            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))

            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            image_np = np.array(image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            overlay = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Original", use_container_width=True)

            with col2:
                st.image(overlay, caption="GradCAM Heatmap", use_container_width=True)