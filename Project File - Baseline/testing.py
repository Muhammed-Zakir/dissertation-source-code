# Run testing
print("\n" + "="*60)
print("TESTING BEST MODEL")
print("="*60)

predictions, targets, probabilities, test_loss, accuracy = test_model_with_probabilities(
    model, test_loader, device
)

output_file_path = os.path.join(drive_dir, 'test_results.txt')

with open(output_file_path, 'w') as f:
    def print_and_save(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, file=f, **kwargs)

    print_and_save(f"\nTest Results:")
    print_and_save(f" Test Loss: {test_loss:.4f}")
    print_and_save(f" Test Accuracy: {accuracy:.2f}%")

    class_names = test_dataset.classes
    n_classes = len(class_names)

    print_and_save(f"\nAUC-ROC Metrics:")

    targets_binarized = label_binarize(targets, classes=range(n_classes))

    try:
        auc_macro = roc_auc_score(targets_binarized, probabilities, multi_class='ovr', average='macro')
        print_and_save(f" Macro-average AUC-ROC: {auc_macro:.4f}")
    except Exception as e:
        print_and_save(f" Could not calculate macro-average AUC: {e}")

    try:
        auc_weighted = roc_auc_score(targets_binarized, probabilities, multi_class='ovr', average='weighted')
        print_and_save(f"   Weighted-average AUC-ROC: {auc_weighted:.4f}")
    except Exception as e:
        print_and_save(f"   Could not calculate weighted-average AUC: {e}")

    print_and_save(f"\n📈 Per-class AUC-ROC scores:")
    for i, class_name in enumerate(class_names):
        try:
            if np.sum(targets_binarized[:, i]) > 0:  # Check if class exists in test set
                class_auc = roc_auc_score(targets_binarized[:, i], probabilities[:, i])
                print_and_save(f"   {class_name}: {class_auc:.4f}")
            else:
                print_and_save(f"   {class_name}: N/A (no samples in test set)")
        except Exception as e:
            print_and_save(f"   {class_name}: Error calculating AUC - {e}")
            
    print_and_save(f"\n📋 Detailed Classification Report:")
    print_and_save(classification_report(targets, predictions, target_names=class_names, digits=4))

    print_and_save(f"\n🎯 Per-class Accuracies:")
    for i, class_name in enumerate(class_names):
        class_mask = np.array(targets) == i
        if np.sum(class_mask) > 0:
            class_accuracy = 100.0 * np.sum((np.array(predictions)[class_mask] == np.array(targets)[class_mask])) / np.sum(class_mask)
            print_and_save(f"   {class_name}: {class_accuracy:.2f}% ({np.sum(class_mask)} samples)")

    plt.figure(figsize=(10, 8)) # Create a new figure for ROC curves

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])

    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        if i < min(8, n_classes):  # Limit to first 8 classes for readability
            try:
                if np.sum(targets_binarized[:, i]) > 0:
                    fpr, tpr, _ = roc_curve(targets_binarized[:, i], probabilities[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=color, lw=2,
                            label=f'{class_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                print_and_save(f"Could not plot ROC for {class_name}: {e}")


    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves - Test Set (First 8 classes)')
    plt.legend(loc="lower right", fontsize='small')


    roc_plot_path = os.path.join(drive_checkpoint_dir, 'test_roc_curves.png')
    plt.savefig(roc_plot_path)
    print_and_save(f"Saved ROC plot to {roc_plot_path}")
    plt.show() # Display the plot


    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(targets, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_plot_path = os.path.join(drive_dir, 'test_confusion_matrix.png')
    plt.savefig(cm_plot_path)
    print_and_save(f"Saved Confusion Matrix plot to {cm_plot_path}")
    plt.show()

    plt.figure(figsize=(10, 8)) 
    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(targets, probabilities[:, 1])
        ap_score = average_precision_score(targets, probabilities[:, 1])
        plt.plot(recall, precision, color='darkorange', lw=2,
                 label=f'PR curve (AP = {ap_score:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Test Set')
        plt.legend(loc="lower left")
    else:
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink'])

        for i, (color, class_name) in enumerate(zip(colors, class_names)):
            if i < min(6, n_classes):  # Limit to first 6 classes for readability
                try:
                    if np.sum(targets_binarized[:, i]) > 0:
                        precision, recall, _ = precision_recall_curve(targets_binarized[:, i], probabilities[:, i])
                        ap_score = average_precision_score(targets_binarized[:, i], probabilities[:, i])
                        plt.plot(recall, precision, color=color, lw=2,
                                label=f'{class_name} (AP = {ap_score:.3f})')
                except Exception as e:
                    print_and_save(f"Could not plot PR curve for {class_name}: {e}")

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Test Set (First 6 classes)')
        plt.legend(loc="lower left", fontsize='small')

    pr_plot_path = os.path.join(drive_dir, 'test_precision_recall_curves.png')
    plt.savefig(pr_plot_path)
    print_and_save(f"Saved Precision-Recall plot to {pr_plot_path}")
    plt.show() # Display the plot


    # Calculate and display average precision scores
    print_and_save(f"\nAverage Precision (AP) Scores:")
    if n_classes == 2:
        ap_score = average_precision_score(targets, probabilities[:, 1])
        print_and_save(f"   Binary Average Precision: {ap_score:.4f}")
    else:
        try:
            ap_macro = average_precision_score(targets_binarized, probabilities, average='macro')
            ap_weighted = average_precision_score(targets_binarized, probabilities, average='weighted')
            print_and_save(f"   Macro-average AP: {ap_macro:.4f}")
            print_and_save(f"   Weighted-average AP: {ap_weighted:.4f}")
        except Exception as e:
            print_and_save(f"   Could not calculate average precision: {e}")

    # Additional statistics
    print_and_save(f"\nAdditional Statistics:")
    print_and_save(f"   Total test samples: {len(targets)}")
    print_and_save(f"   Correct predictions: {sum(np.array(predictions) == np.array(targets))}")
    print_and_save(f"   Incorrect predictions: {sum(np.array(predictions) != np.array(targets))}")
    print_and_save(f"   Number of classes: {n_classes}")

    # Model confidence analysis
    confidence_scores = np.max(probabilities, axis=1)
    print_and_save(f"\nModel Confidence Analysis:")
    print_and_save(f"   Average confidence: {np.mean(confidence_scores):.4f}")
    print_and_save(f"   Std confidence: {np.std(confidence_scores):.4f}")
    print_and_save(f"   Min confidence: {np.min(confidence_scores):.4f}")
    print_and_save(f"   Max confidence: {np.max(confidence_scores):.4f}")

    # Confidence vs Accuracy analysis
    correct_predictions_mask = np.array(predictions) == np.array(targets)
    correct_confidence = confidence_scores[correct_predictions_mask]
    incorrect_confidence = confidence_scores[~correct_predictions_mask]

    if len(incorrect_confidence) > 0:
        print_and_save(f"   Avg confidence (correct): {np.mean(correct_confidence):.4f}")
        print_and_save(f"   Avg confidence (incorrect): {np.mean(incorrect_confidence):.4f}")
        confidence_gap = np.mean(correct_confidence) - np.mean(incorrect_confidence)
        print_and_save(f"   Confidence gap: {confidence_gap:.4f}")

    # Compare with validation performance
    print_and_save(f"\nPerformance Comparison:")
    # You might need to load the best validation accuracy saved during training
    # For now, using the value from the output:
    best_val_accuracy = 0.8900 # Replace with actual loaded value if available
    print_and_save(f"   Best Validation Accuracy: {best_val_accuracy:.4f}%")
    print_and_save(f"   Test Accuracy: {accuracy:.2f}%")
    if accuracy < best_val_accuracy:
        print_and_save(f"   Gap: {best_val_accuracy - accuracy:.2f} percentage points (test lower)")
    else:
        print_and_save(f"   Gap: {accuracy - best_val_accuracy:.2f} percentage points (test higher)")


    print_and_save("\n" + "="*60)
    print_and_save("Testing completed!")
    print_and_save("="*60)

print(f"\nTest results also saved to {output_file_path}")