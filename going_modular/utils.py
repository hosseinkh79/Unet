import torch
import numpy as np


# def intersection_over_union_multiclass(predicted, target):
#     # Assuming predicted and target have shapes (batch_size, num_classes, width, height)

#     intersection = torch.sum(predicted * target)
#     union = torch.sum(predicted) + torch.sum(target) - intersection
    
#     iou = (intersection + 1e-15) / (union + 1e-15)  # Adding a small epsilon to avoid division by zero
    
#     return iou

def compute_iou(predictions, targets, num_classes):
    iou_per_class = np.zeros(num_classes, dtype=np.float32)

    for i in range(num_classes):
        true_positive = np.sum((predictions == i) & (targets == i))
        false_positive = np.sum((predictions == i) & (targets != i))
        false_negative = np.sum((predictions != i) & (targets == i))

        union = true_positive + false_positive + false_negative
        iou = true_positive / union if union > 0 else 0.0
        iou_per_class[i] = iou

    mean_iou = np.mean(iou_per_class)
    return mean_iou


import matplotlib.pyplot as plt
import torch
import os


def plot_loss_curves(results):

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_iou"]
    test_accuracy = results["test_iou"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(10, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_iou")
    plt.plot(epochs, test_accuracy, label="test_iou")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()