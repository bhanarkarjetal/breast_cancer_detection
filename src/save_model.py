import torch


def save_data(
    model,
    optimizer,
    model_state_dict_path: str,
    optimizer_state_dict_path: str,
) -> None:
    """
    Saves the model and optimizer states to the specified paths.

    Args:
        model: The model to be saved.
        optimizer: The optimizer to be saved.
        model_state_dict_path (str): The file path where the model state dict will be saved.
        optimizer_state_dict_path (str): The file path where the optimizer state dict will be saved.
    """
    torch.save(model.state_dict(), model_state_dict_path)

    torch.save(optimizer.state_dict(), optimizer_state_dict_path)


# ==========================================================================================================


def save_model(model, path: str) -> None:
    """
    Saves the entire model to the specified path.

    Args:
        model: The model to be saved.
        path (str): The file path where the model will be saved in .pt format.
    """
    torch.save(model, path)


# ==========================================================================================================


def save_checkpoint(epoch, model, loss, optimizer) -> None:
    """
    Saves the model and optimizer states to a checkpoint file.

    Args:
        model: The model to be saved.
        optimizer: The optimizer to be saved.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"{epoch}_checkpoint.tar",
    )
