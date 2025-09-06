import torch
import torch.nn.functional as F
import torch.nn as nn

def fixmatch_loss(labeled_outputs, labeled_labels, unlabeled_weak_outputs, unlabeled_strong_outputs,
                  supervised_criterion, unsupervised_weight=1.0, confidence_threshold=0.95, 
                  temperature=1.0, epoch=None, total_epochs=None):
    """
    FixMatch loss function with enhanced features.
    
    Args:
        labeled_outputs: Model outputs for labeled data
        labeled_labels: Ground truth labels for labeled data
        unlabeled_weak_outputs: Model outputs for weakly augmented unlabeled data
        unlabeled_strong_outputs: Model outputs for strongly augmented unlabeled data
        supervised_criterion: Loss function for supervised learning
        unsupervised_weight: Weight for unsupervised loss (lambda_u)
        confidence_threshold: Threshold for pseudo-label confidence (tau)
        temperature: Temperature for pseudo-labeling softmax
        epoch: Current epoch (for potential ramp-up)
        total_epochs: Total epochs (for potential ramp-up)
    
    Returns:
        Dictionary containing all loss components and metrics
    """
    
    # Determine device
    if labeled_outputs is not None and labeled_outputs.numel() > 0:
        device = labeled_outputs.device
    elif unlabeled_strong_outputs is not None and unlabeled_strong_outputs.numel() > 0:
        device = unlabeled_strong_outputs.device
    else:
        device = torch.device('cpu')
    
    # Initialize losses and metrics
    supervised_loss = torch.tensor(0.0, device=device, requires_grad=True)
    unsupervised_loss = torch.tensor(0.0, device=device, requires_grad=True)
    mask_ratio = torch.tensor(0.0, device=device)
    
    # 1. Supervised Loss on Labeled Data
    if labeled_outputs is not None and labeled_outputs.numel() > 0 and labeled_labels is not None:
        supervised_loss = supervised_criterion(labeled_outputs, labeled_labels)
    
    # 2. Unsupervised Loss on Unlabeled Data
    if (unlabeled_weak_outputs is not None and unlabeled_weak_outputs.numel() > 0 and
        unlabeled_strong_outputs is not None and unlabeled_strong_outputs.numel() > 0):
        
        # Pseudo-labels from weakly augmented outputs
        with torch.no_grad():
            # Temperature scaling for better calibration
            pseudo_label_logits = unlabeled_weak_outputs / temperature
            pseudo_label_probs = torch.softmax(pseudo_label_logits, dim=1)
            max_probs, pseudo_labels = torch.max(pseudo_label_probs, dim=1)
            
            # Creating the confidence mask
            mask = max_probs >= confidence_threshold
            mask_ratio = mask.float().mean()
        
        # Calculating unsupervised loss only for masked samples
        if mask.sum() > 0:
            unlabeled_strong_masked = unlabeled_strong_outputs[mask]
            pseudo_labels_masked = pseudo_labels[mask]
            unsupervised_loss = F.cross_entropy(unlabeled_strong_masked, pseudo_labels_masked)
    
    # 4. Combine Losses
    combined_loss = supervised_loss + unsupervised_weight * unsupervised_loss
    
    # Dictionary for better handling
    return {
        'total_loss': combined_loss,
        'supervised_loss': supervised_loss,
        'unsupervised_loss': unsupervised_loss,
        'mask_ratio': mask_ratio,
        'current_unsupervised_weight': unsupervised_weight
    }

print("Enhanced FixMatch loss function defined.")