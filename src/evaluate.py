from torchmetrics.classification import (BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score)
import torch

def evaluate_model(preds, true_labels):
    """
    Evaluate the model's performance using various metrics.
    Args:
        preds (torch.Tensor): Predicted binary labels.
        true_labels (torch.Tensor): True binary labels.
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = BinaryAccuracy()(preds, true_labels).item()
    precision = BinaryPrecision()(preds, true_labels).item()
    recall = BinaryRecall()(preds, true_labels).item()
    f1_score = BinaryF1Score()(preds, true_labels).item()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }