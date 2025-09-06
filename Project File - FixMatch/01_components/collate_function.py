# Take the batch and collate (to assemble in proper order) it. 
def fixmatch_collate_fn(batch):
    labeled_images = []
    labeled_labels = []
    unlabeled_weak_views = []
    unlabeled_strong_views = []
    
    for item in batch:
        if item is None:
            continue
        
        if len(item) == 2 and isinstance(item[1], torch.Tensor) and item[1].ndim == 0: # Check for label tensor
            # Labeled data: (image, label)
            labeled_images.append(item[0])
            labeled_labels.append(item[1])
        elif len(item) == 2 and isinstance(item[0], torch.Tensor) and isinstance(item[1], torch.Tensor): # Check for two tensors
            # Unlabeled data: (weak_view, strong_view)
            unlabeled_weak_views.append(item[0])
            unlabeled_strong_views.append(item[1])
        else:
             print(f"Warning: Skipping unexpected item format in batch: {item}")
           
             
    # Stack lists into tensors
    labeled_images = torch.stack(labeled_images) if labeled_images else torch.tensor([])
    labeled_labels = torch.stack(labeled_labels) if labeled_labels else torch.tensor([], dtype=torch.long)
    unlabeled_weak_views = torch.stack(unlabeled_weak_views) if unlabeled_weak_views else torch.tensor([])
    unlabeled_strong_views = torch.stack(unlabeled_strong_views) if unlabeled_strong_views else torch.tensor([])
    
    return labeled_images, labeled_labels, unlabeled_weak_views, unlabeled_strong_views
