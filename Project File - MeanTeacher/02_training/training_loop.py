supervised_criterion = nn.CrossEntropyLoss()

# EMA momentum
ema_momentum = 0.999 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model.to(device)
teacher_model.to(device)

optimizer = optim.Adam(student_model.parameters(), lr=1e-4, weight_decay=1e-3)

# Cosine annealing scheduler
num_epochs_mt = 50

# Learning rate scheduler
scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs_mt,
    eta_min=1e-6
)


drive_checkpoint_dir_mt = "/content/drive/MyDrive/FixMatch Code & Data/Model Checkpoints/meanteacher"
os.makedirs(drive_checkpoint_dir_mt, exist_ok=True)
checkpoint_dir_mt = drive_checkpoint_dir_mt

# Logging setup for Mean Teacher training
log_file_mt = os.path.join(checkpoint_dir_mt, 'mean_teacher_training.log')
with open(log_file_mt, 'w') as f:
    f.write("Epoch,Phase,Batch_or_Summary,CombinedLoss,SupervisedLoss,UnsupervisedLoss,Accuracy,F1_Score,LearningRate,EMAMomentum\n")

print(f"Starting Mean Teacher training for {num_epochs_mt} epochs...")

# Early Stopping and Best Model variables
best_validation_metric = -1.0
epochs_without_improvement = 0
early_stopping_patience = 7

# Initialize lists to store metrics for plotting
train_combined_losses = []
train_supervised_losses = []
train_unsupervised_losses = []
val_losses = []
val_accuracies = []
val_f1_scores = []
epochs_list = []

# --- Training Loop ---
for epoch in range(num_epochs_mt):
    # --- Training Phase ---
    student_model.train()

    total_combined_loss = 0
    total_supervised_loss = 0
    total_unsupervised_loss = 0
    start_time = time.time()

    if 'mean_teacher_dataloader' in locals():
        for batch_idx, (labeled_imgs, labeled_lbls, unlabeled_student, unlabeled_teacher) in enumerate(mean_teacher_dataloader):
            # Move data to device
            labeled_imgs = labeled_imgs.to(device)
            labeled_lbls = labeled_lbls.to(device)
            unlabeled_student = unlabeled_student.to(device)
            unlabeled_teacher = unlabeled_teacher.to(device)

            # Zero the gradients for the student model
            optimizer.zero_grad()

            # Forward Pass
            student_labeled_outputs = student_model(labeled_imgs) if labeled_imgs.numel() > 0 else torch.tensor([])

            # Outputs from student model for unlabeled data
            student_unlabeled_outputs = student_model(unlabeled_student) if unlabeled_student.numel() > 0 else torch.tensor([])

            # Outputs from teacher model for unlabeled data
            with torch.no_grad():
                    teacher_unlabeled_outputs = teacher_model(unlabeled_teacher) if unlabeled_teacher.numel() > 0 else torch.tensor([])


            # Calculate Combined Loss
            combined_loss, supervised_loss, unsupervised_loss = mean_teacher_loss(
                student_labeled_outputs, labeled_lbls,
                student_unlabeled_outputs, teacher_unlabeled_outputs,
                supervised_criterion=supervised_criterion,
                unsupervised_weight=1.0
            )

            # Backpropagation for the student model
            combined_loss.backward()

            # Optimizer step
            optimizer.step()

            # Update teacher network using EMA
            with torch.no_grad():
                for param_teacher, param_student in zip(teacher_model.parameters(), student_model.parameters()):
                    param_teacher.data = ema_momentum * param_teacher.data + (1 - ema_momentum) * param_student.data

            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()

            total_combined_loss += combined_loss.item()
            total_supervised_loss += supervised_loss.item()
            total_unsupervised_loss += unsupervised_loss.item()
            
            # Logging batch information periodically every 50 batches
            if (batch_idx + 1) % 50 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"MT Epoch [{epoch+1}/{num_epochs_mt}], Batch [{batch_idx+1}/{len(mean_teacher_dataloader)}], Combined Loss: {combined_loss.item():.4f}, Sup Loss: {supervised_loss.item():.4f}, Unsup Loss: {unsupervised_loss.item():.4f}, LR: {current_lr:.6f}, EMA Momentum: {ema_momentum:.4f}")
                    with open(log_file_mt, 'a') as f:
                        # Added empty fields for Accuracy and F1 for training batches
                        f.write(f"{epoch+1},Train,{batch_idx+1},{combined_loss.item():.4f},{supervised_loss.item():.4f},{unsupervised_loss.item():.4f},,,{current_lr:.6f},{ema_momentum:.4f}\n")

        # Calculate average losses for the epoch
        avg_combined_loss_train = total_combined_loss / len(mean_teacher_dataloader)
        avg_supervised_loss_train = total_supervised_loss / len(mean_teacher_dataloader)
        avg_unsupervised_loss_train = total_unsupervised_loss / len(mean_teacher_dataloader)

        # Store training metrics for plotting
        train_combined_losses.append(avg_combined_loss_train)
        train_supervised_losses.append(avg_supervised_loss_train)
        train_unsupervised_losses.append(avg_unsupervised_loss_train)

        epoch_time = time.time() - start_time

        print(f"MT Train Epoch [{epoch+1}/{num_epochs_mt}] completed. Avg Combined Loss: {avg_combined_loss_train:.4f}, Avg Sup Loss: {avg_supervised_loss_train:.4f}, Avg Unsup Loss: {avg_unsupervised_loss_train:.4f}, Time: {epoch_time:.2f}s")
        with open(log_file_mt, 'a') as f:
            current_lr = optimizer.param_groups[0]['lr']
            # Log epoch summary (without batch index)
            f.write(f"{epoch+1},Train,EpochSummary,{avg_combined_loss_train:.4f},{avg_supervised_loss_train:.4f},{avg_unsupervised_loss_train:.4f},,,{current_lr:.6f},{ema_momentum:.4f}\n")


    # Validation Phase
    student_model.eval() # Evaluate the student model

    total_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    total_loss_eval = 0

    
    with torch.no_grad():
        for batch_idx_eval, (images_eval, labels_eval) in enumerate(val_dataloader):
            if images_eval is None or labels_eval is None:
                    continue

            images_eval = images_eval.to(device)
            labels_eval = labels_eval.to(device)

            # Forward pass
            outputs_eval = student_model(images_eval)

            # Calculate loss
            loss_eval = supervised_criterion(outputs_eval, labels_eval)
            total_loss_eval += loss_eval.item()

            # Get predictions
            _, predicted = torch.max(outputs_eval.data, 1)

            # Update counts
            total_samples += labels_eval.size(0)
            total_correct += (predicted == labels_eval).sum().item()

            # Store labels and predictions for metrics
            all_labels.extend(labels_eval.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss_eval = total_loss_eval / len(val_dataloader) if len(val_dataloader) > 0 else 0

    # Store validation metrics for plotting
    val_losses.append(avg_loss_eval)
    val_accuracies.append(accuracy)
    epochs_list.append(epoch + 1)

    print(f"MT Eval Epoch [{epoch+1}/{num_epochs_mt}]. Loss: {avg_loss_eval:.4f}, Accuracy: {accuracy:.4f}")
    with open(log_file_mt, 'a') as f:
        # Log evaluation metrics
        # Added empty fields for Unsupervised Loss and Mask Ratio for eval summary
        f.write(f"{epoch+1},Eval,EpochSummary,{avg_loss_eval:.4f},,{avg_unsupervised_loss_train:.4f},{accuracy:.4f},,\n") # Using training unsupervised loss for logging consistency, or leave blank


    # Checkpointing and Early Stopping
    if accuracy > best_validation_metric:
        best_validation_metric = accuracy
        epochs_without_improvement = 0 # Reset counter

        checkpoint_path_best = os.path.join(checkpoint_dir_mt, 'best_mean_teacher_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'student_model_state_dict': student_model.state_dict(),
            'teacher_model_state_dict': teacher_model.state_dict(), # Save teacher state as well
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'train_combined_loss': avg_combined_loss_train, # Saving training loss from the epoch
            'eval_loss': avg_loss_eval,
            'accuracy': accuracy # Saving evaluation accuracy
        }, checkpoint_path_best)
        print(f"Saved best model checkpoint to colab with Accuracy: {accuracy:.4f}")
        
        drive_checkpoint_path_best = os.path.join(drive_checkpoint_dir_mt, 'best_mean_teacher_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'student_model_state_dict': student_model.state_dict(),
            'teacher_model_state_dict': teacher_model.state_dict(), # Save teacher state as well
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'train_combined_loss': avg_combined_loss_train, # Saving training loss from the epoch
            'eval_loss': avg_loss_eval,
            'accuracy': accuracy # Saving evaluation accuracy
        }, drive_checkpoint_path_best)
        print(f"Saved best model checkpoint to drive with Accuracy: {accuracy:.4f}")
    else:
        epochs_without_improvement += 1
        print(f"Validation accuracy did not improve. Patience: {epochs_without_improvement}/{early_stopping_patience}")

    # Check for early stopping
    if epochs_without_improvement >= early_stopping_patience:
        print(f"Early stopping triggered after {epoch+1} epochs due to no improvement in validation accuracy for {early_stopping_patience} epochs.")
        break # Exit the training loop

print("\nMean Teacher training finished.")


# Loading best checkpoint for evaluation
if os.path.exists(os.path.join(checkpoint_dir_mt, 'best_mean_teacher_model.pth')):
    print("\n--- Final Best Model Metrics (from checkpoint) ---")
    best_checkpoint = torch.load(os.path.join(checkpoint_dir_mt, 'best_mean_teacher_model.pth'))
    print(f"Best Epoch: {best_checkpoint['epoch']}")
    print(f"Best Evaluation Accuracy: {best_checkpoint['accuracy']:.4f}")
    print(f"Best Evaluation F1 Score (weighted): {best_checkpoint['f1_score']:.4f}")