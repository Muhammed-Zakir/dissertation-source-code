
# Set GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Google Drive for Storage
drive.mount('/content/drive')

checkpoint_dir = '/content/drive/MyDrive/FixMatch Code & Data/Model Checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
    
# Verify Checkpoint Directory
test_file_path = os.path.join(checkpoint_dir, 'test_file.txt')
with open(test_file_path, 'w') as f:
    f.write('This is a test file.')
print(f"Test file created at: {test_file_path}")

# Augmentations
weak_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

strong_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)), # translate 5% of image width/height
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.0, hue=0.0),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.15, 1.8)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03 * x.std())
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])


# Import Dataset 
import kagglehub
path = kagglehub.dataset_download("muhammedzakir/semi-supervised-brain-tumor-dataset-cropped")
print("Path to dataset files:", path)

# Set Dataset Directory
dataset_dir = path
labeled_data_dir = os.path.join(dataset_dir, 'Training')
unlabeled_data_dir = os.path.join(dataset_dir, 'Unlabeled_Training')
validation_data_dir = os.path.join(dataset_dir, 'Validation')

fixmatch_batch_size = 32 
unlabeled_ratio = 3 # 3 unlabeled for every 1 labeled

# Calculate expected labeled and unlabeled samples per batch
num_labeled_per_batch = fixmatch_batch_size // (1 + unlabeled_ratio)
num_unlabeled_per_batch = num_labeled_per_batch * unlabeled_ratio

# Instantiate the Training Dataset 
fixmatch_dataset = FixMatchDataset(
    labeled_data_dir=labeled_data_dir,
    unlabeled_data_dir=unlabeled_data_dir,
    weak_transform=weak_transform,
    strong_transform=strong_transform
)

# Instantiate Validation Dataset
val_dataset = ValidationDataset(
    data_dir=validation_data_dir,
    transform=val_transforms
)

# Instantiate Batch Sampler
fixmatch_batch_sampler = FixMatchBatchSampler(
    dataset=fixmatch_dataset,
    batch_size=fixmatch_batch_size,
    unlabeled_ratio=unlabeled_ratio,
    num_iterations=None # Or specify a fixed number of iterations per epoch
)
        
# Instantiate Training DataLoader
fixmatch_dataloader = DataLoader(
    fixmatch_dataset,
    batch_sampler=fixmatch_batch_sampler, 
    num_workers=2,
    collate_fn=fixmatch_collate_fn
)


# Verify A Batch
print("\nSample batch shapes (requires data to be present):")
try:
    labeled_imgs, labeled_lbls, unlabeled_weak, unlabeled_strong = next(iter(fixmatch_dataloader))
    print(f"Labeled Images shape: {labeled_imgs.shape}, Labeled Labels shape: {labeled_lbls.shape}")
    print(f"Unlabeled Weak Views shape: {unlabeled_weak.shape}, Unlabeled Strong Views shape: {unlabeled_strong.shape}")
except Exception as e:
    print(f"Could not retrieve sample from FixMatch dataloader: {e}")

    
# Instantiate Validation Dataloader
val_loader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False,
    num_workers=2
)


# ResNet50 Encoder with Imagenet Weights, Final Classification Layer Removed
resnet50_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet50_encoder = nn.Sequential(*list(resnet50_encoder.children())[:-1])
pretrained_encoder = resnet50_encoder
num_classes = 4


# Instantiate fixmatch model
fixmatch_model = FixMatchModel(pretrained_encoder, num_classes)
print("FixMatch Model architecture defined and instantiated.")

# Print model architecture
print(f"\nModel Architecture:")
print(fixmatch_model)