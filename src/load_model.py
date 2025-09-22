import torch


def load_model(
    model,
    optimizer,
    model_state_dict_path: str,
    optimizer_state_dict_path: str,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Loads the model and optimizer states from the specified paths.

    Args:
        model: The model to be loaded.
        optimizer: The optimizer to be loaded.
        model_state_dict_path (str): The file path from where the model state dict will be loaded.
        optimizer_state_dict_path (str): The file path from where the optimizer state dict will be loaded.

    Returns:
        model: The model with loaded state dict.
        optimizer: The optimizer with loaded state dict.
    """
    model.load_state_dict(torch.load(model_state_dict_path))
    optimizer.load_state_dict(torch.load(optimizer_state_dict_path))

    return model, optimizer


def load_entire_model(path: str) -> torch.nn.Module:
    """
    Loads the entire model from the specified path.

    Args:
        path (str): The file path from where the model will be loaded in .pt format.

    Returns:
        model: The loaded model.
    """
    model = torch.load(path)
    return model


def load_model_from_checkpoint(
    model, optimizer, checkpoint_path: str
) -> tuple[torch.nn.Module, torch.optim.Optimizer, float, int]:
    """
    Loads the model and optimizer states from a checkpoint file.

    Args:
        model: The model to be loaded.
        optimizer: The optimizer to be loaded.
        checkpoint_path (str): The file path from where the checkpoint will be loaded.
        model_state_dict_path (str): The file path from where the model state dict will be loaded.
        optimizer_state_dict_path (str): The file path from where the optimizer state dict will be loaded.

    Returns:
        model: The model with loaded state dict.
        optimizer: The optimizer with loaded state dict.
        loss: The loss value at the time of saving the checkpoint.
        epoch: The epoch number at the time of saving the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss = checkpoint["loss"]
    epoch = checkpoint["epoch"]

    print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")

    return model, optimizer, loss, epoch
