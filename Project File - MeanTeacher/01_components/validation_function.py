
# Validation function
def validate_model(model, val_loader, criterion, device):
    """
    Evaluate model on validation set.
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
        class_accuracies: Per-class accuracies
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 4  # For 4 classes
    class_total = [0] * 4
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    # Per-class accuracies
    class_accuracies = []
    for i in range(4):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    
    model.train()  # Set back to training mode
    return avg_loss, accuracy, class_accuracies