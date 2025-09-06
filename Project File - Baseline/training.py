import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os

# training log
log_dir = os.path.join(drive_dir, "training_logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

criterion = nn.CrossEntropyLoss() # Supervised loss criterian
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

epochs = 50
patience = 5
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
train_acc = []
val_acc = []

print("Starting training...")
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    logging.info(f"Epoch {epoch+1}/{epochs}")

    # Training phase
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / len(train_loader.dataset)
    epoch_train_acc = correct_predictions / total_predictions
    train_losses.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)

    # Validation phase
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_val_loss = running_loss / len(val_loader.dataset)
    epoch_val_acc = correct_predictions / total_predictions
    val_losses.append(epoch_val_loss)
    val_acc.append(epoch_val_acc)

    print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
    logging.info(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    # Early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        # Save the best model checkpoint to Google Drive
        checkpoint_path = os.path.join(drive_dir, 'transfer_learning_exp_best_model.pth')
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Saved best model checkpoint to {checkpoint_path}.")

    else:
        patience_counter += 1
        print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}") # Print patience counter
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.") # Added epoch number
            logging.info(f"Early stopping triggered at epoch {epoch+1}.") # Added epoch number
            break

    # Learning rate scheduling
    scheduler.step(epoch_val_loss)

print("Training finished.")


# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
# Save the loss plot to Google Drive
loss_plot_path = os.path.join(drive_checkpoint_dir, 'training_validation_loss.png')
plt.savefig(loss_plot_path)
print(f"Saved loss plot to {loss_plot_path}")
plt.show()

# Plot accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
# Save the accuracy plot to Google Drive
accuracy_plot_path = os.path.join(drive_checkpoint_dir, 'training_validation_accuracy.png')
plt.savefig(accuracy_plot_path)
print(f"Saved accuracy plot to {accuracy_plot_path}")
plt.show()