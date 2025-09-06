class MeanTeacherDataset(Dataset):
    """
    Custom Dataset for Mean Teacher Semi-Supervised Learning.

    Loads images and labels from labeled and unlabeled directories,
    applying distinct augmentation pipelines for labeled data (student input)
    and for unlabeled data (generating two different random views for student and teacher inputs).
    """
    def __init__(self, labeled_data_dir, unlabeled_data_dir,
                 labeled_transform=None, unlabeled_transform=None):
        """
        Args:
            labeled_data_dir (str): Path to the directory containing labeled data (with class subdirectories).
            unlabeled_data_dir (str): Path to the directory containing unlabeled data.
            labeled_transform (torchvision.transforms.Compose, optional): Transform for labeled images (student input).
            unlabeled_transform (torchvision.transforms.Compose, optional): Transform that returns TWO DIFFERENT
                                                                           random augmented views of the SAME image
                                                                           for unlabeled data (student and teacher inputs).
        """
        self.labeled_data_dir = labeled_data_dir
        self.unlabeled_data_dir = unlabeled_data_dir
        self.labeled_transform = labeled_transform
        self.unlabeled_transform = unlabeled_transform # This should return (student_view, teacher_view)

        self.labeled_image_paths = []
        self.labeled_labels = []
        self.unlabeled_image_paths = []
        self.class_to_idx = {}

        # Collect labeled image paths and labels
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


        # Collect unlabeled image paths
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
        """Returns the total number of images (labeled + unlabeled)."""
        return self.total_images

    def __getitem__(self, idx):
        """
        Loads and augments an image.

        Returns:
            tuple: For labeled data: (augmented_image (torch.Tensor), label (int)).
                   For unlabeled data: (student_view (torch.Tensor), teacher_view (torch.Tensor), -1 (int)).
        """
        # Determine if the index corresponds to labeled or unlabeled data
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
            # Apply labeled transform (student input)
            if self.labeled_transform:
                augmented_image = self.labeled_transform(image)
                return augmented_image, torch.tensor(label, dtype=torch.long)
            else:
                 return transforms.ToTensor()(image), torch.tensor(label, dtype=torch.long)

        else: # Unlabeled data
            # Apply unlabeled transform which returns two different views
            if self.unlabeled_transform:
                 student_view, teacher_view = self.unlabeled_transform(image)
                 return student_view, teacher_view, torch.tensor(label, dtype=torch.long)
            else:
                 # If no unlabeled transform, return two identical ToTensor views
                 img_tensor = transforms.ToTensor()(image)
                 return img_tensor, img_tensor, torch.tensor(label, dtype=torch.long)