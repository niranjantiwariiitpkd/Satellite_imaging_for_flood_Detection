import torch
from sklearn.metrics import classification_report, confusion_matrix
from dataset import val_loader
from model import WaterClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

model = WaterClassifier().to(device)
model.load_state_dict(torch.load("../models/water_model.pth"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

        all_preds.extend(preds.flatten())
        all_labels.extend(labels.numpy())

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))