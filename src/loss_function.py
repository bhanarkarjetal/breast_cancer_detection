import torch
import torch.nn as nn
from typing import Optional

def compute_loss(predictions: torch.Tensor, 
                 targets: torch.Tensor,
                 class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the cross-entropy loss between predictions and targets.
    
    Args:
        predictions (torch.Tensor): The predicted logits from the model.
        targets (torch.Tensor): The true class labels.
    
    Returns:
        torch.Tensor: The computed cross-entropy loss.
        """
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    return criterion(predictions, targets)