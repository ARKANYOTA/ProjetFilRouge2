import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    """
    Classic LeNet-5 architecture adapted for flexible input channels and classes.
    """
    def __init__(self, in_channels=1, num_classes=2, image_size=(64, 64)):
        super(LeNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2) # Keeps size if input is 28x28 or 32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dry run to get fc input size and register fc1 properly
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size[0], image_size[1])
            out = self.pool1(F.relu(self.conv1(dummy)))
            out = self.pool2(F.relu(self.conv2(out)))
            self.fc_input_dim = out.view(1, -1).size(1)
            
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Apply convolutions and pooling
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool2(F.relu(self.conv2(out)))
        
        # Flatten
        out = out.view(out.size(0), -1)
        
        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
