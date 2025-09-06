import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model evaluation
downstream_model.eval()

running_loss = 0.0
running_corrects = 0
total_samples = 0
all_labels_downstream = []
all_preds_downstream = []

with torch.no_grad():
    for inputs, labels in tqdm(test_dataloader, desc="Downstream Testing"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = downstream_model(inputs)
        loss = criterion_downstream(outputs, labels)
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += inputs.size(0)
        
        all_labels_downstream.extend(labels.cpu().numpy())
        all_preds_downstream.extend(preds.cpu().numpy())

test_loss_downstream = running_loss / total_samples
test_acc_downstream = running_corrects.double() / total_samples

print(f"Downstream Test Loss: {test_loss_downstream:.4f}")
print(f"Downstream Test Accuracy: {test_acc_downstream.item():.4f}")

# Calculate classification metrics
if len(all_labels_downstream) > 0:
    all_labels_downstream = np.array(all_labels_downstream)
    all_preds_downstream = np.array(all_preds_downstream)
    
    f1_downstream = f1_score(all_labels_downstream, all_preds_downstream, 
                            average='weighted', zero_division=0)
    precision_downstream = precision_score(all_labels_downstream, all_preds_downstream, 
                                          average='weighted', zero_division=0)
    recall_downstream = recall_score(all_labels_downstream, all_preds_downstream, 
                                    average='weighted', zero_division=0)
    
    print(f"Downstream F1 Score (weighted): {f1_downstream:.4f}")
    print(f"Downstream Precision (weighted): {precision_downstream:.4f}")
    print(f"Downstream Recall (weighted): {recall_downstream:.4f}")
    
    # Confusion Matrix
    conf_matrix_downstream = confusion_matrix(all_labels_downstream, all_preds_downstream)
    print("Downstream Confusion Matrix:")
    print(conf_matrix_downstream)
    
    # Get class names
    if hasattr(test_dataloader.dataset, 'classes'):
        class_names = test_dataloader.dataset.classes
    elif hasattr(test_dataloader.dataset, 'dataset') and hasattr(test_dataloader.dataset.dataset, 'classes'):
        class_names = test_dataloader.dataset.dataset.classes
    else:
        class_names = [f'Class {i}' for i in range(len(np.unique(all_labels_downstream)))]
    
    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_downstream,
                                display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title("Downstream Confusion Matrix", fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification Report
    print("Downstream Classification Report:")
    print(classification_report(all_labels_downstream, all_preds_downstream, 
                              target_names=class_names, zero_division=0))
else:
    print("No predictions or labels collected for downstream testing.")