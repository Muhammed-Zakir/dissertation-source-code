def mean_teacher_collate_fn(batch):
    labeled_images = []
    labeled_labels = []
    unlabeled_student_views = []
    unlabeled_teacher_views = []
    unlabeled_labels = []

    for item in batch:
        if item is None:
            continue

        # Check if it's labeled data (tuple of size 2, second element is a tensor label)
        if len(item) == 2 and isinstance(item[1], torch.Tensor) and item[1].ndim == 0 and item[1] != -1:
             labeled_images.append(item[0])
             labeled_labels.append(item[1])
        # Check if it's unlabeled data (tuple of size 3, third element is -1)
        elif len(item) == 3 and isinstance(item[2], torch.Tensor) and item[2].ndim == 0 and item[2] == -1:
             unlabeled_student_views.append(item[0])
             unlabeled_teacher_views.append(item[1])
             unlabeled_labels.append(item[2])
        else:
             print(f"Warning: Skipping unexpected item format in batch: {item}")


    # Stack lists into tensors
    labeled_images = torch.stack(labeled_images) if labeled_images else torch.tensor([])
    labeled_labels = torch.stack(labeled_labels) if labeled_labels else torch.tensor([], dtype=torch.long)
    unlabeled_student_views = torch.stack(unlabeled_student_views) if unlabeled_student_views else torch.tensor([])
    unlabeled_teacher_views = torch.stack(unlabeled_teacher_views) if unlabeled_teacher_views else torch.tensor([])

    # Return separate batches for labeled and unlabeled data
    return labeled_images, labeled_labels, unlabeled_student_views, unlabeled_teacher_views