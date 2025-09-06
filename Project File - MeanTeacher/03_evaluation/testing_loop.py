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

# Run testing
print("\n" + "="*60)
print("TESTING BEST MODEL")
print("="*60)

# Get predictions, targets, and probabilities
predictions, targets, probabilities, test_loss, accuracy = test_model_with_probabilities(
    student_model, test_loader, device
)

print(f"\nTest Results:")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {accuracy:.2f}%")

# Get class names and number of classes
class_names = test_dataset.classes
n_classes = len(class_names)

# Calculate AUC-ROC metrics
print(f"\nAUC-ROC Metrics:")

# Multi-class classification
targets_binarized = label_binarize(targets, classes=range(n_classes))
    
# Calculate macro-average AUC
auc_macro = roc_auc_score(targets_binarized, probabilities, multi_class='ovr', average='macro')
print(f"   Macro-average AUC-ROC: {auc_macro:.4f}")


# Calculate weighted-average AUC
auc_weighted = roc_auc_score(targets_binarized, probabilities, multi_class='ovr', average='weighted')
print(f"   Weighted-average AUC-ROC: {auc_weighted:.4f}")


# Per-class AUC scores
print(f"\nPer-class AUC-ROC scores:")
for i, class_name in enumerate(class_names):
    try:
        if np.sum(targets_binarized[:, i]) > 0:  # Check if class exists in test set
            class_auc = roc_auc_score(targets_binarized[:, i], probabilities[:, i])
            print(f"   {class_name}: {class_auc:.4f}")
        else:
            print(f"   {class_name}: N/A (no samples in test set)")
    except Exception as e:
        print(f"   {class_name}: Error calculating AUC - {e}")

# Detailed classification report
print(f"\nDetailed Classification Report:")
print(classification_report(targets, predictions, target_names=class_names, digits=4))

# Per-class accuracy
print(f"\nPer-class Accuracies:")
for i, class_name in enumerate(class_names):
    class_mask = np.array(targets) == i
    if np.sum(class_mask) > 0:
        class_accuracy = 100.0 * np.sum((np.array(predictions)[class_mask] == np.array(targets)[class_mask])) / np.sum(class_mask)
        print(f"   {class_name}: {class_accuracy:.2f}% ({np.sum(class_mask)} samples)")

# Plot ROC curves
plt.figure(figsize=(15, 5))

# Multi-class ROC curves
plt.subplot(1, 3, 1)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])

for i, (color, class_name) in enumerate(zip(colors, class_names)):
    if i < min(4, n_classes):
        try:
            if np.sum(targets_binarized[:, i]) > 0:
                fpr, tpr, _ = roc_curve(targets_binarized[:, i], probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
        except Exception as e:
            print(f"Could not plot ROC for {class_name}: {e}")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves (First 8 classes)')
plt.legend(loc="lower right", fontsize='small')

# Confusion Matrix
plt.subplot(1, 3, 2)
cm = confusion_matrix(targets, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Precision-Recall curves
plt.subplot(1, 3, 3)
if n_classes == 2:
    precision, recall, _ = precision_recall_curve(targets, probabilities[:, 1])
    ap_score = average_precision_score(targets, probabilities[:, 1])
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AP = {ap_score:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
else:
    # Multi-class average precision
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])
    
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        if i < min(6, n_classes):  # Limit to first 6 classes for readability
            try:
                if np.sum(targets_binarized[:, i]) > 0:
                    precision, recall, _ = precision_recall_curve(targets_binarized[:, i], probabilities[:, i])
                    ap_score = average_precision_score(targets_binarized[:, i], probabilities[:, i])
                    plt.plot(recall, precision, color=color, lw=2,
                            label=f'{class_name} (AP = {ap_score:.3f})')
            except Exception as e:
                print(f"Could not plot PR curve for {class_name}: {e}")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (First 6 classes)')
    plt.legend(loc="lower left", fontsize='small')

plt.tight_layout()
plt.show()

# Calculate and display average precision scores
print(f"\nAverage Precision (AP) Scores:")
ap_macro = average_precision_score(targets_binarized, probabilities, average='macro')
ap_weighted = average_precision_score(targets_binarized, probabilities, average='weighted')
print(f"   Macro-average AP: {ap_macro:.4f}")
print(f"   Weighted-average AP: {ap_weighted:.4f}")


# Additional statistics
print(f"\nAdditional Statistics:")
print(f"   Total test samples: {len(targets)}")
print(f"   Correct predictions: {sum(np.array(predictions) == np.array(targets))}")
print(f"   Incorrect predictions: {sum(np.array(predictions) != np.array(targets))}")
print(f"   Number of classes: {n_classes}")

# Model confidence analysis
confidence_scores = np.max(probabilities, axis=1)
print(f"\nModel Confidence Analysis:")
print(f"   Average confidence: {np.mean(confidence_scores):.4f}")
print(f"   Std confidence: {np.std(confidence_scores):.4f}")
print(f"   Min confidence: {np.min(confidence_scores):.4f}")
print(f"   Max confidence: {np.max(confidence_scores):.4f}")

# Confidence vs Accuracy analysis
correct_predictions = np.array(predictions) == np.array(targets)
correct_confidence = confidence_scores[correct_predictions]
incorrect_confidence = confidence_scores[~correct_predictions]

if len(incorrect_confidence) > 0:
    print(f"   Avg confidence (correct): {np.mean(correct_confidence):.4f}")
    print(f"   Avg confidence (incorrect): {np.mean(incorrect_confidence):.4f}")
    confidence_gap = np.mean(correct_confidence) - np.mean(incorrect_confidence)
    print(f"   Confidence gap: {confidence_gap:.4f}")

print("\n" + "="*60)
print("Testing completed!")
print("="*60)