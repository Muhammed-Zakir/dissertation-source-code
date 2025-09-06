import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import random
import os
import time
import copy
from tqdm import tqdm

# Functions
def save_checkpoint(state, checkpoint_path):
    torch.save(state, checkpoint_path)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading and preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

cleaned_data_dir = "cleaned_dataset"
image_dataset = datasets.ImageFolder(root=cleaned_data_dir, transform=data_transforms)

print("Original unique class labels:", sorted(set(image_dataset.targets)))

# Class label mapping
unique_classes = sorted(set(image_dataset.targets))
class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
print("Class mapping:", class_mapping)

image_dataset.targets = [class_mapping[t] for t in image_dataset.targets]
print("New class labels:", sorted(set(image_dataset.targets)))

# Class balancing
desired_count_per_class = 500
class_indices = {i: [] for i in range(len(unique_classes))}
for i, target in enumerate(image_dataset.targets):
    class_indices[target].append(i)

balanced_indices = []
for class_id, indices in class_indices.items():
    num_samples = len(indices)
    if num_samples > desired_count_per_class:
        balanced_indices.extend(random.sample(indices, desired_count_per_class))
    else:
        balanced_indices.extend(random.choices(indices, k=desired_count_per_class))

balanced_dataset = Subset(image_dataset, balanced_indices)

# Dataset splitting
subset_size = int(len(balanced_dataset) * 0.5)
subset_indices = random.sample(range(len(balanced_dataset)), subset_size)
subset_dataset = Subset(balanced_dataset, subset_indices)

subset_labels = [balanced_dataset[i][1] for i in subset_indices]

train_indices, temp_indices, train_labels, temp_labels = train_test_split(
    range(len(subset_dataset)), subset_labels, 
    test_size=0.2, stratify=subset_labels, random_state=42
)
val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, stratify=[subset_labels[i] for i in temp_indices], random_state=42
)

train_dataset = Subset(subset_dataset, train_indices)
val_dataset = Subset(subset_dataset, val_indices)
test_dataset = Subset(subset_dataset, test_indices)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")
print(f"Test size: {len(test_dataset)}")

# Data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))

# Model architecture
resnet = models.resnet50(weights=None)
backbone = nn.Sequential(*list(resnet.children())[:-1])

# Load pretrained SSL weights
pretrained_path = "/kaggle/input/resnet50_byol_pretext_final_model/pytorch/default/1/final_ssl_model.pth"

if os.path.exists(pretrained_path):
    state_dict = torch.load(pretrained_path, map_location=device)
    
    backbone_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('0.'):
            backbone_state_dict[k[2:]] = v
    
    backbone.load_state_dict(backbone_state_dict, strict=False)
    print(f"Loaded SSL pretrained backbone weights from {pretrained_path}")
else:
    print(f"SSL pretrained model checkpoint not found at {pretrained_path}")

# Freeze backbone parameters
for param in backbone.parameters():
    param.requires_grad = False

print("Backbone parameters frozen.")

# Classification head
num_classes = len(image_dataset.classes)

classification_head = nn.Sequential(
    nn.Linear(resnet.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

# Complete downstream model
downstream_model = nn.Sequential(
    backbone,
    nn.Flatten(),
    classification_head
)

downstream_model = downstream_model.to(device)
print("Downstream model for classification defined.")

# Training setup
criterion_downstream = nn.CrossEntropyLoss()
optimizer_downstream = optim.Adam(downstream_model.parameters(), lr=0.001)
scheduler_downstream = ReduceLROnPlateau(optimizer_downstream, mode='min', factor=0.5, patience=2, verbose=True)
print("Loss function and optimizer defined.")

# Training parameters
num_downstream_epochs = 50
patience_downstream = 5

best_downstream_model_wts = copy.deepcopy(downstream_model.state_dict())
best_downstream_acc = 0.0
epochs_without_improvement_downstream = 0

downstream_checkpoint_dir = "downstream_model_checkpoints"
os.makedirs(downstream_checkpoint_dir, exist_ok=True)

downstream_train_losses = []
downstream_train_accuracies = []
downstream_val_losses = []
downstream_val_accuracies = []

print("Starting downstream training...")

# Training loop
for epoch in range(num_downstream_epochs):
    epoch_start_time = time.time()
    
    # Training phase
    downstream_model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for inputs, labels in tqdm(train_dataloader, desc=f"Downstream Train Epoch {epoch + 1}"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer_downstream.zero_grad()
        
        with torch.set_grad_enabled(True):
            outputs = downstream_model(inputs)
            loss = criterion_downstream(outputs, labels)
            
            loss.backward()
            optimizer_downstream.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
    
    epoch_train_loss = running_loss / total_samples
    epoch_train_acc = running_corrects.double() / total_samples
    downstream_train_losses.append(epoch_train_loss)
    downstream_train_accuracies.append(epoch_train_acc.item())
    
    # Validation phase
    downstream_model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_dataloader, desc=f"Downstream Val Epoch {epoch + 1}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = downstream_model(inputs)
            loss = criterion_downstream(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
    
    epoch_val_loss = running_loss / total_samples
    epoch_val_acc = running_corrects.double() / total_samples
    downstream_val_losses.append(epoch_val_loss)
    downstream_val_accuracies.append(epoch_val_acc.item())
    scheduler_downstream.step(epoch_val_loss)
    current_lr = optimizer_downstream.param_groups[0]['lr']
    
    print(f"Downstream Epoch {epoch+1}/{num_downstream_epochs}")
    print(f"Train Loss: {epoch_train_loss:.4f} Train Acc: {epoch_train_acc:.4f}")
    print(f"Val Loss: {epoch_val_loss:.4f} Val Acc: {epoch_val_acc:.4f}")
    print("-" * 20)
    print(f"Current learning rate: {current_lr:.6f}")
    
    # Save best model
    if epoch_val_acc > best_downstream_acc:
        best_downstream_acc = epoch_val_acc
        best_downstream_model_wts = copy.deepcopy(downstream_model.state_dict())
        epochs_without_improvement_downstream = 0
        checkpoint_path = os.path.join(downstream_checkpoint_dir, f"best_downstream_model_epoch_{epoch + 1}.pth")
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': downstream_model.state_dict(),
            'optimizer_state_dict': optimizer_downstream.state_dict(),
            'scheduler_state_dict': scheduler_downstream.state_dict(),
            'train_loss': loss
        }, checkpoint_path=checkpoint_path)
        print(f"Saved best downstream model checkpoint to {checkpoint_path}")
    else:
        epochs_without_improvement_downstream += 1
        print(f"Downstream validation accuracy did not improve for {epochs_without_improvement_downstream} epoch(s).")
    
    # Early stopping
    if epochs_without_improvement_downstream >= patience_downstream:
        print(f"Early stopping after {epoch+1} epochs due to no improvement for {patience_downstream} consecutive epochs.")
        break
    
    epoch_end_time = time.time()
    print(f"Epoch time: {(epoch_end_time - epoch_start_time):.2f} seconds")

print("Downstream training finished.")

# Load best model weights
downstream_model.load_state_dict(best_downstream_model_wts)
print("Loaded best downstream model weights.")