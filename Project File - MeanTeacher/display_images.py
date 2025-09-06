import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Get a batch of images from the dataloader
try:
    labeled_imgs, labeled_lbls, unlabeled_student, unlabeled_teacher = next(iter(mean_teacher_dataloader))

    # Display the labeled images
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Labeled Images")
    plt.imshow(vutils.make_grid(labeled_imgs, padding=2, normalize=True).permute(1, 2, 0))
    plt.show()

    # Display the unlabeled student views
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Unlabeled Student Views")
    plt.imshow(vutils.make_grid(unlabeled_student, padding=2, normalize=True).permute(1, 2, 0))
    plt.show()

    # Display the unlabeled teacher views
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Unlabeled Teacher Views")
    plt.imshow(vutils.make_grid(unlabeled_teacher, padding=2, normalize=True).permute(1, 2, 0))
    plt.show()

except NameError:
    print("Error: 'mean_teacher_dataloader' is not defined. Cannot display images.")
except Exception as e:
    print(f"An error occurred while displaying images: {e}")