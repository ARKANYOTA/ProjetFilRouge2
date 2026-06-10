import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaborWaveletFilterBank(nn.Module):
    """
    Custom Gabor Filter Bank to mock a 2D Scattering Transform.
    Applies fixed Gabor wavelets at different scales and orientations,
    followed by a modulus (absolute value) and average pooling.
    """
    def __init__(self, in_channels=4, scales=2, orientations=4, kernel_size=9):
        super(GaborWaveletFilterBank, self).__init__()
        self.in_channels = in_channels
        self.scales = scales
        self.orientations = orientations
        self.kernel_size = kernel_size
        
        # Precompute Gabor kernels
        kernels = []
        for scale in range(scales):
            # Scale defines the frequency (lambda) and size (sigma)
            wavelength = 3.0 * (2.0 ** scale)
            sigma = 2.0 * (2.0 ** scale)
            for ori in range(orientations):
                theta = ori * np.pi / orientations
                kernel = self._get_gabor_kernel(kernel_size, wavelength, theta, sigma)
                kernels.append(kernel)
                
        # Shape: (scales * orientations, 1, kernel_size, kernel_size)
        kernels_tensor = torch.stack(kernels).unsqueeze(1) # Add channel dim for depthwise conv
        self.register_buffer('gabor_kernels', kernels_tensor)
        
    def _get_gabor_kernel(self, size, wavelength, theta, sigma, gamma=1.0, psi=0.0):
        """Generates a 2D Gabor filter kernel."""
        half_size = size // 2
        y, x = np.meshgrid(
            np.arange(-half_size, half_size + 1),
            np.arange(-half_size, half_size + 1)
        )
        
        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        # Gabor formula
        gb = np.exp(-.5 * (x_theta**2 + (gamma * y_theta)**2) / sigma**2) * np.cos(2 * np.pi * x_theta / wavelength + psi)
        return torch.tensor(gb, dtype=torch.float32)
        
    def forward(self, x):
        # Input shape: (Batch, in_channels, H, W)
        batch_size, c, h, w = x.shape
        
        # Reshape to treat channels as batch dim for depthwise-like Gabor filtering
        x_reshaped = x.view(batch_size * c, 1, h, w)
        
        # Apply Gabor convolution (padding to keep size)
        pad = self.kernel_size // 2
        # conv_out shape: (Batch * c, scales * orientations, H, W)
        conv_out = F.conv2d(x_reshaped, self.gabor_kernels, padding=pad)
        
        # Modulus: non-linear rectification (absolute value)
        # This acts as the Scattering transform modulus operator
        mod_out = torch.abs(conv_out)
        
        # Low-pass average pooling to get translation invariance
        # pool_out shape: (Batch * c, scales * orientations, H // 4, W // 4)
        pool_out = F.avg_pool2d(mod_out, kernel_size=4, stride=4)
        
        # Reconstruct batch and channel dimensions
        scat_channels = self.scales * self.orientations
        out = pool_out.view(batch_size, c * scat_channels, pool_out.shape[2], pool_out.shape[3])
        return out

class ScatteringCNN(nn.Module):
    """
    Deep Hybrid Network wrapping Gabor Scattering Wavelets.
    Paper: "Scaling the Scattering Transform: Deep Hybrid Networks" (ICCV 2017)
    """
    def __init__(self, in_channels=4, num_classes=8, image_size=(64, 64)):
        super(ScatteringCNN, self).__init__()
        scales = 2
        orientations = 4
        
        # Fixed Scattering Wavelet stage
        self.scat_transform = GaborWaveletFilterBank(
            in_channels=in_channels, 
            scales=scales, 
            orientations=orientations
        )
        
        # Input dimension to learned layers:
        # Each input channel is scattered into scales * orientations channels.
        # For 4 input channels, scales=2, orientations=4 -> 4 * 8 = 32 channels.
        scat_out_channels = in_channels * scales * orientations
        scat_h = image_size[0] // 4
        scat_w = image_size[1] // 4
        
        # Learned Hybrid CNN Stage
        # Typically uses 1x1 convolutions to compress/mix the scattering coefficients,
        # followed by standard conv/pooling and fully connected layers.
        self.conv1 = nn.Conv2d(scat_out_channels, 64, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # size down to H//8, W//8
        
        self.fc_input_dim = 128 * (scat_h // 2) * (scat_w // 2)
        
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # 1. Fixed Gabor Scattering feature extraction
        x = self.scat_transform(x)
            
        # 2. Learned Convolutional mixing
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 3. Dense classification
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
