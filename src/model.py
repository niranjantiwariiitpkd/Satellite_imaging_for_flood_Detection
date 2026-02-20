import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class WaterClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)