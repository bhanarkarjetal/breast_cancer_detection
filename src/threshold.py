import torch
import numpy as np
from torchmetrics.classification import BinaryF1Score

def get_best_thres(probs, true_labels):
    """
    Calculate the optimal threshold for binary classification based on predictions and true labels.

    Args:
        preds (torch.Tensor): Predicted probabilities (after sigmoid) of shape (N,).
        true_label (torch.Tensor): True binary labels of shape (N,).
    Returns:
        float: Optimal threshold value.
    """

    best_thres = 0.5
    best_f1 = 0.0

    for thres in torch.linspace(0.0, 1.0, steps=101):
        preds = (probs >= thres).int()

        f1 = BinaryF1Score(threshold = 0.5)(preds, true_labels).item()
        if f1 > best_f1:
            best_f1 = f1
            best_thres = np.round(thres,2)

    return best_thres