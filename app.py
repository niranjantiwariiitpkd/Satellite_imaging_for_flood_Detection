import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.model import WaterClassifier

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = WaterClassifier().to(device)
model.load_state_dict(torch.load("models/water_model.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.set_page_config(page_title="Flood Risk Detection", layout="centered")

st.title("🌊 Satellite Flood Risk Assessment System")
st.write("Upload a satellite image to estimate flood risk based on water detection.")

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probability = torch.sigmoid(output).item()

    # Display Probability
    st.subheader("Water Detection Probability")
    st.progress(probability)
    st.write(f"{probability*100:.2f}%")

    # Risk Logic
    if probability < 0.3:
        st.success("🟢 Low Flood Risk")
    elif probability < 0.6:
        st.info("🟡 Moderate Flood Risk")
    elif probability < 0.8:
        st.warning("🟠 High Flood Risk")
    else:
        st.error("🔴 Severe Flood Risk")

    st.markdown("---")
    st.write("### Model Details")
    st.write("""
    - Architecture: ResNet18 (Transfer Learning)
    - Dataset: EuroSAT Satellite Dataset
    - Task: Flood(water) vs Nonflood Classification
    - Application: Flood Risk Estimation
    """)