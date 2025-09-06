# Mount Google Drive
from google.colab import drive
import os
import kagglehub
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

drive.mount('/content/drive')

# Download the dataset
path = kagglehub.dataset_download("muhammedzakir/semi-supervised-brain-tumor-dataset-cropped")
print("Path to dataset files:", path)

dataset_dir = path
training_data_dir = os.path.join(dataset_dir, 'Training')
validation_data_dir = os.path.join(dataset_dir, 'Validation')

# Drive directory for storage
drive_dir = ''
os.makedirs(drive_dir, exist_ok=True)

# datasets
train_dataset = ImageFolder(root=training_data_dir, transform=transforms)
val_dataset = ImageFolder(root=validation_data_dir, transform=transforms)

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ResNet50 architecture
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

num_ftrs = model.fc.in_features
num_classes = 4 
# replaced the classification layer with 4 classes
model.fc = nn.Sequential(
    nn.Dropout(0.5), 
    nn.Linear(num_ftrs, num_classes)
)

# Move the model to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)



