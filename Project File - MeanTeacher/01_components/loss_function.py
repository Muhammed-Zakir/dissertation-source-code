def mean_teacher_loss(student_labeled_outputs, labeled_labels,
                      student_unlabeled_outputs, teacher_unlabeled_outputs,
                      supervised_criterion, unsupervised_weight=1.0):
    """
    Calculates the combined Mean Teacher loss.

    Args:
        student_labeled_outputs (torch.Tensor): Student model outputs (logits) for labeled data.
        labeled_labels (torch.Tensor): True labels for labeled data.
        student_unlabeled_outputs (torch.Tensor): Student model outputs (logits) for unlabeled data.
        teacher_unlabeled_outputs (torch.Tensor): Teacher model outputs (logits) for unlabeled data.
        supervised_criterion (torch.nn.Module): The supervised loss function (e.g., CrossEntropyLoss).
        unsupervised_weight (float): Weighting factor for the unsupervised loss.

    Returns:
        torch.Tensor: The combined Mean Teacher loss.
        torch.Tensor: The supervised loss component.
        torch.Tensor: The unsupervised loss component.
    """
    # 1. Supervised Loss on Labeled Data (using student predictions)
    # Only calculate if there is labeled data in the batch
    if student_labeled_outputs.numel() > 0:
        supervised_loss = supervised_criterion(student_labeled_outputs, labeled_labels)
    else:
        # Ensure device is compatible with other tensors if no labeled data
        device = student_unlabeled_outputs.device if student_unlabeled_outputs.numel() > 0 else (teacher_unlabeled_outputs.device if teacher_unlabeled_outputs.numel() > 0 else torch.device("cpu"))
        supervised_loss = torch.tensor(0.0).to(device)


    # 2. Unsupervised Loss on Unlabeled Data (Consistency Loss)
    # Encourage consistency between student and teacher predictions for unlabeled data
    # Mean Teacher often uses Mean Squared Error between predicted probability distributions
    # Or KL Divergence. Let's use MSE on logits for simplicity here.

    unsupervised_loss = torch.tensor(0.0).to(student_unlabeled_outputs.device if student_unlabeled_outputs.numel() > 0 else (teacher_unlabeled_outputs.device if teacher_unlabeled_outputs.numel() > 0 else torch.device("cpu")))

    if student_unlabeled_outputs.numel() > 0 and teacher_unlabeled_outputs.numel() > 0:
        # Detach teacher outputs as targets
        teacher_unlabeled_outputs_detached = teacher_unlabeled_outputs.detach()

        # Calculate MSE between student and teacher logits
        unsupervised_loss = F.mse_loss(student_unlabeled_outputs, teacher_unlabeled_outputs_detached)

    # 3. Combine Losses
    combined_loss = supervised_loss + unsupervised_weight * unsupervised_loss

    return combined_loss, supervised_loss, unsupervised_loss

print("Mean Teacher loss function defined.")