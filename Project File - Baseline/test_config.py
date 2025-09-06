import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import torch
import torch.nn.functional as F

def test_model_with_probabilities(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()

            # Get probabilities using softmax
            probabilities = F.softmax(outputs, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * sum(np.array(all_predictions) == np.array(all_targets)) / len(all_targets)

    return all_predictions, all_targets, np.array(all_probabilities), avg_loss, accuracy



model.load_state_dict(torch.load(os.path.join(drive_dir, 'transfer_learning_exp_best_model.pth')))
model.to(device)

dataset_dir = path
test_data_dir = os.path.join(dataset_dir, 'Testing')

best_model_path = 'semi_supervised_checkpoints/best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
                                     
test_dataset = datasets.ImageFolder(
    root=test_data_dir,
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)