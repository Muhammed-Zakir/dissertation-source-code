class FixMatchDataset(Dataset):
    def __init__(self, labeled_data_dir, unlabeled_data_dir,
                 weak_transform=None, strong_transform=None):
        # image directories and transformations
        self.labeled_data_dir = labeled_data_dir
        self.unlabeled_data_dir = unlabeled_data_dir
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        
        # image paths, corresponding labels for labeled images, class as index 
        self.labeled_image_paths = []
        self.labeled_labels = []
        self.unlabeled_image_paths = []
        self.class_to_idx = {}
        
        # labeled image paths and labels
        if os.path.exists(labeled_data_dir):
            class_names = sorted(os.listdir(labeled_data_dir))
            for i, class_name in enumerate(class_names):
                class_dir = os.path.join(labeled_data_dir, class_name)
                if os.path.isdir(class_dir):
                    self.class_to_idx[class_name] = i
                    for filename in os.listdir(class_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.labeled_image_paths.append(os.path.join(class_dir, filename))
                            self.labeled_labels.append(i)
        else:
            print(f"Warning: Labeled data directory not found at {labeled_data_dir}. Proceeding without labeled data.")

        # unlabeled image paths
        if os.path.exists(unlabeled_data_dir):
             for root, _, files in os.walk(unlabeled_data_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.unlabeled_image_paths.append(os.path.join(root, file))
        else:
             print(f"Warning: Unlabeled data directory not found at {unlabeled_data_dir}. Proceeding without unlabeled data.")
             
        
        self.total_labeled = len(self.labeled_image_paths)
        self.total_unlabeled = len(self.unlabeled_image_paths)
        self.total_images = self.total_labeled + self.total_unlabeled

        if self.total_images == 0:
            raise FileNotFoundError("No image files found in either labeled or unlabeled data directories.")

        print(f"Found {self.total_labeled} labeled images.")
        print(f"Found {self.total_unlabeled} unlabeled images.")
        
        
    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        # Index correlates with labeled or unlabeled count
        if idx < self.total_labeled:
            # Load labeled data
            image_path = self.labeled_image_paths[idx]
            label = self.labeled_labels[idx]
            is_labeled = True
        else:
            # Load unlabeled data
            image_path = self.unlabeled_image_paths[idx - self.total_labeled]
            label = -1 # Indicator for unlabeled data
            is_labeled = False

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        if is_labeled:
            # Apply weak augmentation to labeled data
            if self.weak_transform:
                augmented_image = self.weak_transform(image)
                return augmented_image, torch.tensor(label, dtype=torch.long)
            else:
                 # Return original image as tensor if no weak transform
                 return transforms.ToTensor()(image), torch.tensor(label, dtype=torch.long)

        else: # Unlabeled data
            # Apply BOTH weak and strong augmentations
            weak_view = self.weak_transform(image) if self.weak_transform else transforms.ToTensor()(image)
            strong_view = self.strong_transform(image) if self.strong_transform else transforms.ToTensor()(image)
            # Return both views
            return weak_view, strong_view
