from typing import Dict, Any
import torch
import torch.optim as optim

def create_optimizer(
        model: torch.nn.Module,
        optimizer_name: str = 'adam',
        optimizer_params: Dict[str, Any] = None) -> torch.optim.Optimizer:
    """
    Returns an optimizer for the given model.
    
    Args:
        model (torch.nn.Module): The neural network model.
        optimizer_name (str): The name of the optimizer to use.
        optimizer_params (dict, optional): Additional parameters for the optimizer.

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), **(optimizer_params or {}))
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), **(optimizer_params or {}))
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


