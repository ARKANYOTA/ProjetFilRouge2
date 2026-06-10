import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np
import os
import csv
from PIL import Image

class BiasedDataset(Dataset):
    """
    Wrapper Dataset that introduces a bias variable S and a noise channel epsilon.
    
    Parameters:
    -----------
    base_dataset : Dataset
        The original dataset returning (image, label).
    p0 : float
        Probability P(S=1 | y=0).
    p1 : float
        Probability P(S=1 | y=1).
    """
    def __init__(self, base_dataset, p0=0.5, p1=0.5):
        self.base_dataset = base_dataset
        self.p0 = p0
        self.p1 = p1
        
        self.S_labels = []
        for _, label in self.base_dataset:
            # For multi-class (0-7), we treat label=1 as class 1 and all other labels as class 0 for S generation
            p = self.p1 if label == 1 else self.p0
            s = int(np.random.binomial(1, p))
            self.S_labels.append(s)
            
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        s = self.S_labels[idx]
        
        c, h, w = img.shape
        
        if s == 0:
            epsilon = torch.zeros((1, h, w), dtype=torch.float32)
        else:
            epsilon = torch.randn((1, h, w), dtype=torch.float32)
            
        # Concatenate epsilon as a new channel
        # Resulting shape: (C+1, H, W)
        biased_img = torch.cat([img, epsilon], dim=0)
        
        return biased_img, label, s

class KaggleTildaDataset(Dataset):
    """
    Dataset to load flat TIF images from the Kaggle dataset using train.csv labels.
    """
    def __init__(self, csv_file, img_dir, transform=None, binary_classes=None):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.transform = transform
        self.binary_classes = binary_classes
        
        self.samples = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader) # Skip header (id;label)
            for row in reader:
                if not row:
                    continue
                img_id, label = row[0], int(row[1])
                
                # If binary mode, filter classes
                if binary_classes is not None:
                    if label in binary_classes:
                        mapped_label = binary_classes.index(label)
                        self.samples.append((img_id, mapped_label))
                else:
                    self.samples.append((img_id, label))
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        
        # Load image
        img = Image.open(img_path)
        img = img.convert('RGB') # Convert to RGB
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label

def get_tilda_transforms(image_size=(64, 64), augment=False):
    """
    Returns standard transforms for the dataset.
    Question 2.1: Pre-process images and data augmentation.
    """
    transform_list = []
    if augment:
        transform_list.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ])
    else:
        transform_list.append(transforms.Resize(image_size))
        
    transform_list.extend([
        transforms.ToTensor(),
    ])
    
    return transforms.Compose(transform_list)

def load_tilda_dataset(data_dir, image_size=(64, 64), p0=0.5, p1=0.5, augment=False, binary_classes=None):
    """
    Helper function to load the dataset from a directory.
    If 'train.csv' is present, uses custom KaggleTildaDataset.
    Otherwise, defaults to ImageFolder (e.g. for synthetic dummy data).
    """
    transform = get_tilda_transforms(image_size, augment=augment)
    
    csv_file = os.path.join(data_dir, 'train.csv')
    img_dir = os.path.join(data_dir, 'train')
    
    if os.path.exists(csv_file) and os.path.exists(img_dir):
        base_dataset = KaggleTildaDataset(csv_file, img_dir, transform=transform, binary_classes=binary_classes)
    else:
        base_dataset = ImageFolder(root=data_dir, transform=transform)
        
    # Wrap with our biased dataset
    return BiasedDataset(base_dataset, p0=p0, p1=p1)

