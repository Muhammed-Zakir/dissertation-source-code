import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time

# Supervised criterion
supervised_criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fixmatch_model.to(device)

# Training hyperparameters
num_epochs_ssl = 50
validate_every = 3
early_stopping_patience = 5

# Optimizer
optimizer = optim.Adam(
    fixmatch_model.parameters(), 
    lr=3e-4,
    weight_decay=1e-4
)

# Learning rate scheduler
scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=num_epochs_ssl,
    eta_min=1e-6
)

# Create checkpoint directories
checkpoint_dir_ssl = '/content/drive/MyDrive/FixMatch Code & Data/Model Checkpoints'
colab_checkpoint_dir = "/content/checkpoints"
os.makedirs(checkpoint_dir_ssl, exist_ok=True)
os.makedirs(colab_checkpoint_dir, exist_ok=True)

# Logging setup
log_file_ssl = os.path.join(checkpoint_dir_ssl, 'semi_supervised_training.log')
val_log_file = os.path.join(checkpoint_dir_ssl, 'validation_log.log')

with open(log_file_ssl, 'w') as f:
    f.write("Epoch,Batch,CombinedLoss,SupervisedLoss,UnsupervisedLoss,LearningRate,MaskRatio\n")

with open(val_log_file, 'w') as f:
    f.write("Epoch,ValidationLoss,ValidationAccuracy,Class0Acc,Class1Acc,Class2Acc,Class3Acc\n")

# Early stopping variables
best_val_acc = 0.0
best_model_state = None
epochs_without_improvement = 0

print(f"Starting semi-supervised training with validation for {num_epochs_ssl} epochs...")
print(f"Using device: {device}")
print(f"Validation every {validate_every} epochs, Early stopping patience: {early_stopping_patience}")
print(f"Optimizer: {type(optimizer).__name__}")
print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")

# Training loop
for epoch in range(num_epochs_ssl):
    # Training phase
    fixmatch_model.train()
    total_combined_loss = 0
    total_supervised_loss = 0
    total_unsupervised_loss = 0
    total_mask_ratio = 0
    num_batches = 0
    start_time = time.time()
    
    for batch_idx, batch_data in enumerate(fixmatch_dataloader):
        # Handle different batch formats
        if len(batch_data) == 4:
            labeled_imgs, labeled_lbls, unlabeled_v1, unlabeled_v2 = batch_data
        else:
            print(f"Warning: Unexpected batch format with {len(batch_data)} elements")
            continue
        
        # Move data to device
        labeled_imgs = labeled_imgs.to(device) if labeled_imgs is not None else None
        labeled_lbls = labeled_lbls.to(device) if labeled_lbls is not None else None
        unlabeled_v1 = unlabeled_v1.to(device) if unlabeled_v1 is not None else None
        unlabeled_v2 = unlabeled_v2.to(device) if unlabeled_v2 is not None else None
        
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        labeled_outputs = None
        unlabeled_outputs1 = None
        unlabeled_outputs2 = None

        if labeled_imgs is not None and labeled_imgs.numel() > 0:
            labeled_outputs = fixmatch_model(labeled_imgs)

        if unlabeled_v1 is not None and unlabeled_v1.numel() > 0:
            unlabeled_outputs1 = fixmatch_model(unlabeled_v1)

        if unlabeled_v2 is not None and unlabeled_v2.numel() > 0:
            unlabeled_outputs2 = fixmatch_model(unlabeled_v2)

        # Calculate combined loss
        try:
            loss_dict = fixmatch_loss(
                labeled_outputs, labeled_lbls, 
                unlabeled_outputs1, unlabeled_outputs2,
                supervised_criterion=supervised_criterion,
                unsupervised_weight=1.0,
                temperature=1.0,
                confidence_threshold=0.95,
                epoch=epoch,
                total_epochs=num_epochs_ssl
            )
            
            combined_loss = loss_dict['total_loss']
            supervised_loss = loss_dict['supervised_loss']
            unsupervised_loss = loss_dict['unsupervised_loss']
            mask_ratio = loss_dict['mask_ratio']
            
        except Exception as e:
            print(f"Error in loss calculation at epoch {epoch+1}, batch {batch_idx+1}: {e}")
            continue

        # Skip if loss is invalid
        if torch.isnan(combined_loss) or torch.isinf(combined_loss):
            print(f"Invalid loss detected at epoch {epoch+1}, batch {batch_idx+1}. Skipping batch.")
            continue
        
        # Backpropagation
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(fixmatch_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        total_combined_loss += combined_loss.item()
        total_supervised_loss += supervised_loss.item()
        total_unsupervised_loss += unsupervised_loss.item()
        total_mask_ratio += mask_ratio.item() if isinstance(mask_ratio, torch.Tensor) else mask_ratio
        num_batches += 1
        
        # Log batch information periodically
        if (batch_idx + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            avg_mask = mask_ratio.item() if isinstance(mask_ratio, torch.Tensor) else mask_ratio
            print(f"SSL Epoch [{epoch+1}/{num_epochs_ssl}], Batch [{batch_idx+1}/{len(fixmatch_dataloader)}], "
                    f"Combined Loss: {combined_loss.item():.4f}, Sup Loss: {supervised_loss.item():.4f}, "
                    f"Unsup Loss: {unsupervised_loss.item():.4f}, LR: {current_lr:.6f}, Mask Ratio: {avg_mask:.3f}")
            
            # Log to file
            with open(log_file_ssl, 'a') as f:
                f.write(f"{epoch+1},{batch_idx+1},{combined_loss.item():.4f},"
                        f"{supervised_loss.item():.4f},{unsupervised_loss.item():.4f},"
                        f"{current_lr:.6f},{avg_mask:.3f}\n")
    
    # Calculate epoch averages
    if num_batches > 0:
        avg_combined_loss = total_combined_loss / num_batches
        avg_supervised_loss = total_supervised_loss / num_batches
        avg_unsupervised_loss = total_unsupervised_loss / num_batches
        avg_mask_ratio = total_mask_ratio / num_batches
    else:
        avg_combined_loss = avg_supervised_loss = avg_unsupervised_loss = avg_mask_ratio = 0.0
    
    epoch_time = time.time() - start_time
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print epoch summary
    print(f"SSL Epoch [{epoch+1}/{num_epochs_ssl}] completed. "
            f"Avg Combined Loss: {avg_combined_loss:.4f}, Avg Sup Loss: {avg_supervised_loss:.4f}, "
            f"Avg Unsup Loss: {avg_unsupervised_loss:.4f}, Avg Mask Ratio: {avg_mask_ratio:.3f}, "
            f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
    
    # Update learning rate scheduler
    scheduler.step()
    
    # Validation phase
    if (epoch + 1) % validate_every == 0:
        print(f"\n{'='*60}")
        print(f"VALIDATION at Epoch {epoch+1}")
        print(f"{'='*60}")
        
        val_loss, val_acc, class_accs = validate_model(fixmatch_model, val_loader, supervised_criterion, device)
        
        print(f"Validation Results:")
        print(f"  Loss: {val_loss:.4f}")
        print(f"  Accuracy: {val_acc:.2f}%")
        print(f"  Per-class Accuracies:")
        class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']  # Update with your class names
        for i, (name, acc) in enumerate(zip(class_names, class_accs)):
            print(f"     {name}: {acc:.1f}%")
        
        # Log validation results
        with open(val_log_file, 'a') as f:
            f.write(f"{epoch+1},{val_loss:.4f},{val_acc:.2f},"
                    f"{class_accs[0]:.2f},{class_accs[1]:.2f},{class_accs[2]:.2f},{class_accs[3]:.2f}\n")
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = fixmatch_model.state_dict().copy()
            epochs_without_improvement = 0
            
            print(f"NEW BEST MODEL! Validation accuracy: {val_acc:.2f}%")
            
            # Save best model checkpoint
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_accuracies': class_accs,
                'train_losses': {
                    'combined': avg_combined_loss,
                    'supervised': avg_supervised_loss,
                    'unsupervised': avg_unsupervised_loss,
                    'mask_ratio': avg_mask_ratio
                }
            }
            
            # Save to both locations
            best_colab_path = os.path.join(colab_checkpoint_dir, 'best_model.pth')
            best_drive_path = os.path.join(checkpoint_dir_ssl, 'best_model.pth')
            torch.save(best_checkpoint, best_colab_path)
            torch.save(best_checkpoint, best_drive_path)
            
            print(f"Best model saved to: {best_colab_path}")
            print(f"Best model saved to: {best_drive_path}")
        else:
            epochs_without_improvement += validate_every
            print(f"No improvement. Best: {best_val_acc:.2f}%, Current: {val_acc:.2f}%")
            print(f"Epochs without improvement: {epochs_without_improvement}")
        
        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEARLY STOPPING triggered!")
            print(f"   No improvement for {early_stopping_patience} epochs")
            print(f"   Best validation accuracy: {best_val_acc:.2f}%")
            break
        
        print(f"{'='*60}\n")
    
    # Regular checkpoint saving
    if (epoch + 1) % 10 == 0 or epoch == num_epochs_ssl - 1:
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': fixmatch_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'combined_loss': avg_combined_loss,
            'supervised_loss': avg_supervised_loss,
            'unsupervised_loss': avg_unsupervised_loss,
            'mask_ratio': avg_mask_ratio,
            'hyperparameters': {
                'learning_rate': optimizer.param_groups[0]['lr'],
                'weight_decay': optimizer.param_groups[0]['weight_decay'],
                'num_epochs': num_epochs_ssl,
                'confidence_threshold': 0.95,
                'unsupervised_weight': 1.0
            }
        }
        
        # Save regular checkpoints
        colab_checkpoint_path = os.path.join(colab_checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        drive_checkpoint_path = os.path.join(checkpoint_dir_ssl, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint_data, colab_checkpoint_path)
        torch.save(checkpoint_data, drive_checkpoint_path)
        
        print(f"Checkpoint saved: Epoch {epoch+1}")

# Load best model at the end
if best_model_state is not None:
    fixmatch_model.load_state_dict(best_model_state)
    print(f"\nTRAINING COMPLETED!")
    print(f"   Loaded best model with validation accuracy: {best_val_acc:.2f}%")
    
    # Save final best model
    final_model_path_colab = os.path.join(colab_checkpoint_dir, 'final_best_model.pth')
    final_model_path_drive = os.path.join(checkpoint_dir_ssl, 'final_best_model.pth')
    torch.save(best_model_state, final_model_path_colab)
    torch.save(best_model_state, final_model_path_drive)
    
    print(f"Final best model saved to: {final_model_path_colab}")
    print(f"Final best model saved to: {final_model_path_drive}")
else:
    print(f"\nTraining completed without validation improvements")

print("🎊 FixMatch Semi-supervised training with validation completed!")