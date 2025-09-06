import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import GaussianBlur
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import time
import copy
from tqdm import tqdm
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preprocessing functions
def crop_brain_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 45, 255, cv2.THRESH_BINARY)
    
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    new_img = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    return new_img

def preprocess_folder(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    classes = os.listdir(source_dir)
    for cls in classes:
        class_src = os.path.join(source_dir, cls)
        class_tgt = os.path.join(target_dir, cls)
        os.makedirs(class_tgt, exist_ok=True)
        
        for img_name in os.listdir(class_src):
            img_path = os.path.join(class_src, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            brain = crop_brain_contour(image)
            if brain is None:
                continue
            
            brain = cv2.resize(brain, (256, 256))
            save_path = os.path.join(class_tgt, img_name)
            cv2.imwrite(save_path, brain)
    print(f"Cleaned images saved to: {target_dir}")

def save_checkpoint(state, checkpoint_path):
    torch.save(state, checkpoint_path)

# BYOL augmentation class
class BYOLTransform:
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=45),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        if not isinstance(x, Image.Image):
            x = transforms.ToPILImage()(x)
        return self.transform(x), self.transform(x)

class PairedImageFolder(Dataset):
    def __init__(self, image_folder_dataset, transform=None):
        self.image_folder_dataset = image_folder_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.image_folder_dataset)
    
    def __getitem__(self, idx):
        img, label = self.image_folder_dataset[idx]
        if self.transform:
            view1, view2 = self.transform(img)
            return view1, view2, label
        return img, img, label

# Loss function
def byol_loss(online_pred, target_proj):
    online_pred = F.normalize(online_pred, dim=1)
    target_proj = F.normalize(target_proj, dim=1)
    loss = 2 - 2 * (online_pred * target_proj).sum(dim=1).mean()
    return loss

# Training function
def train_ssl_epoch(online_model, target_network, dataloader, criterion, optimizer, device, momentum):
    online_model.train()
    running_loss = 0.0
    total_samples = 0
    
    for view1, view2, _ in tqdm(dataloader, desc="SSL Training"):
        view1 = view1.to(device)
        view2 = view2.to(device)
        
        optimizer.zero_grad()
        
        online_pred_1 = online_model(view1)
        online_pred_2 = online_model(view2)
        
        with torch.no_grad():
            target_proj_1 = target_network(view1)
            target_proj_2 = target_network(view2)
        
        loss = (criterion(online_pred_1, target_proj_2) + criterion(online_pred_2, target_proj_1)) / 2
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            for param_q, param_k in zip(online_network.parameters(), target_network.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        
        running_loss += loss.item() * view1.size(0)
        total_samples += view1.size(0)
    
    return running_loss / total_samples

# Data preprocessing
data_dir = "/kaggle/input/brats-dataset-classification"
preprocess_folder(data_dir, 'cleaned_dataset')

# Dataset transforms
data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root='cleaned_dataset', transform=data_transforms)

# Create paired dataset and dataloader
byol_transform = BYOLTransform(224)
paired_dataset = PairedImageFolder(full_dataset, transform=byol_transform)
train_dataloader_ssl = DataLoader(paired_dataset, batch_size=32, shuffle=True, num_workers=2)

# Model architecture
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
backbone = nn.Sequential(*list(resnet.children())[:-1])

projection_head = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(inplace=True),
    nn.Linear(2048, 256)
)

prediction_head = nn.Sequential(
    nn.Linear(256, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Linear(512, 256)
)

online_network = nn.Sequential(backbone, nn.Flatten(), projection_head)
online_model = nn.Sequential(online_network, prediction_head)
target_network = copy.deepcopy(online_network)

online_model = online_model.to(device)
target_network = target_network.to(device)

for param in target_network.parameters():
    param.requires_grad = False

# Optimizer and scheduler
optimizer = optim.Adam(online_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Load checkpoint if available
checkpoint_path = "/kaggle/input/resnet50_byol_pretext_final_model/pytorch/default/1/final_ssl_model.pth"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    online_model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")
    start_epoch = 30
else:
    print("No checkpoint found, starting from beginning")
    start_epoch = 0

# Training loop
num_ssl_epochs = 100
initial_momentum = 0.996
final_momentum = 1.0
ssl_train_losses = []

ssl_checkpoint_dir = "ssl_model_checkpoints"
os.makedirs(ssl_checkpoint_dir, exist_ok=True)

print("Starting self-supervised training...")

for epoch in range(start_epoch, num_ssl_epochs):
    epoch_start_time = time.time()
    
    momentum = initial_momentum + (final_momentum - initial_momentum) * epoch / (num_ssl_epochs - 1)
    ssl_loss = train_ssl_epoch(online_model, target_network, train_dataloader_ssl, byol_loss, optimizer, device, momentum)
    ssl_train_losses.append(ssl_loss)
    
    print(f"SSL Epoch {epoch+1}/{num_ssl_epochs}")
    print(f"Train Loss: {ssl_loss:.4f}")
    print("-" * 20)
    
    if (epoch + 1) % 10 == 0 or (epoch + 1) == num_ssl_epochs:
        checkpoint_path = os.path.join(ssl_checkpoint_dir, f"ssl_model_epoch_{epoch + 1}.pth")
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': online_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': ssl_loss
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    epoch_end_time = time.time()
    print(f"Epoch time: {(epoch_end_time - epoch_start_time):.2f} seconds")

print("Self-supervised training finished.")

# Save final model
final_model_path = os.path.join(ssl_checkpoint_dir, "final_ssl_model.pth")
torch.save(online_model.state_dict(), final_model_path)
print(f"Saved final SSL model to {final_model_path}")

# Save and plot loss
df = pd.DataFrame({'Epoch': range(1, len(ssl_train_losses) + 1), 'Loss': ssl_train_losses})
df.to_csv('ssl_train_losses.csv', index=False)

plt.figure(figsize=(8, 5))
plt.plot(df['Epoch']+30, df['Loss'], marker='o', linestyle='-', color='blue')
plt.title('SSL Training Loss Curve BYOL ResNet50')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('ssl_train_loss_curve.png')
plt.show()