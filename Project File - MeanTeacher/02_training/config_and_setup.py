# Dataset directiries
dataset_dir = path
labeled_data_dir = os.path.join(dataset_dir, 'Training')
unlabeled_data_dir = os.path.join(dataset_dir, 'Unlabeled_Training')
validation_data_dir = os.path.join(dataset_dir, 'Validation')


# Instantiate the MeanTeacherDataset
mean_teacher_dataset = MeanTeacherDataset(
    labeled_data_dir=labeled_data_dir,
    unlabeled_data_dir=unlabeled_data_dir,
    labeled_transform=labeled_transform_student, # Use student augmentation for labeled
    unlabeled_transform=TwoDifferentRandomAugmentations(unlabeled_augmentation) # Apply unlabeled_augmentation twice randomly
)

mean_teacher_batch_size = 32
unlabeled_ratio = 3

num_labeled_per_batch = mean_teacher_batch_size // (1 + unlabeled_ratio)
num_unlabeled_per_batch = num_labeled_per_batch * unlabeled_ratio

mean_teacher_batch_sampler = BatchSampler(
    dataset=mean_teacher_dataset,
    batch_size=mean_teacher_batch_size,
    unlabeled_ratio=unlabeled_ratio,
    num_iterations=None
)

# PyTorch DataLoader for the MeanTeacherDataset
mean_teacher_dataloader = DataLoader(
    mean_teacher_dataset,
    batch_sampler=mean_teacher_batch_sampler,
    num_workers=2, # Use multiple workers
    collate_fn=mean_teacher_collate_fn # Use the custom collate function
)

# Verify A batch
print("\nSample batch shapes:")
try:
    labeled_imgs, labeled_lbls, unlabeled_student, unlabeled_teacher = next(iter(mean_teacher_dataloader))
    print(f"Labeled Images shape: {labeled_imgs.shape}, Labeled Labels shape: {labeled_lbls.shape}")
    print(f"Unlabeled Student Views shape: {unlabeled_student.shape}, Unlabeled Teacher Views shape: {unlabeled_teacher.shape}")
except Exception as e:
    print(f"Could not retrieve sample from Mean Teacher dataloader: {e}")

except NameError:
    print("Error: 'mean_teacher_dataset' is not defined. Cannot create dataloader.")
except Exception as e:
print(f"An error occurred during DataLoader setup: {e}")

    
# Backbone encoder
resnet50_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet50_encoder = nn.Sequential(*list(resnet50_encoder.children())[:-1])
pretrained_encoder = resnet50_encoder

num_classes = 4

# Instantiate the Student Model
student_model = BaseClassifier(pretrained_encoder, num_classes, dropout_rate=0.6)
print("Student model architecture defined and instantiated.")

# Instantiate the Teacher Model  
teacher_model = copy.deepcopy(student_model)
print("Teacher model architecture defined and instantiated as a copy of the student.")

# Freeze the teacher model parameters
for param in teacher_model.parameters():
    param.requires_grad = False
print("Teacher model parameters frozen.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model.to(device)
teacher_model.to(device)

# Test the model Instantiation
try:
    # Get logits from the models
    student_labeled_logits = student_model(labeled_imgs) if labeled_imgs.numel() > 0 else torch.tensor([])
    student_unlabeled_logits = student_model(unlabeled_student_views) if unlabeled_student_views.numel() > 0 else torch.tensor([])
    teacher_unlabeled_logits = teacher_model(unlabeled_teacher_views) if unlabeled_teacher_views.numel() > 0 else torch.tensor([])

    supervised_criterion = nn.CrossEntropyLoss()

    total_loss, sup_loss, un_loss = mean_teacher_loss(
        student_labeled_logits, labeled_lbls,
        student_unlabeled_logits, teacher_unlabeled_logits,
        supervised_criterion=supervised_criterion, unsupervised_weight=1.0
    )
    print(f"Example combined loss: {total_loss.item():.4f}, Supervised loss: {sup_loss.item():.4f}, Unsupervised loss: {un_loss.item():.4f}")

except NameError:
    print("\nSkipping example usage as necessary variables are not defined.")
except Exception as e:
    print(f"\nAn error occurred during example loss calculation: {e}")



# Validation Dataset
val_dataset = ValidationDataset(
    data_dir=validation_data_dir,
    transform=validation_transforms
)


val_dataloader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False,  # Important: don't shuffle validation
    num_workers=2
)


#--------------------------------------------------------------------
# Test Configuration

# Dataset
dataset_dir = path
test_data_dir = os.path.join(dataset_dir, 'Testing')
best_model_path = 'semi_supervised_checkpoints/best_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test transforms
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# Load test dataset
test_dataset = datasets.ImageFolder(
    root=test_data_dir,
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

print(f"Test dataset size: {len(test_dataset)}")
print(f"Number of classes: {len(test_dataset.classes)}")
print(f"Classes: {test_dataset.classes}")
