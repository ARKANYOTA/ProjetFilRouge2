import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    """
    ResNet-18 wrapper from torchvision adapted for flexible input channels and classes.
    """
    def __init__(self, in_channels=1, num_classes=2, pretrained=False):
        super(ResNet, self).__init__()
        # Load standard resnet18
        if pretrained:
            # Using new weights API for modern torchvision
            weights = models.ResNet18_Weights.DEFAULT
            self.resnet = models.resnet18(weights=weights)
        else:
            self.resnet = models.resnet18(weights=None)
            
        # Adapt first conv layer if in_channels is not 3
        if in_channels != 3:
            # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 
                64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
            
        # Adapt last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
