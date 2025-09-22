from typing import Dict

import torch

from config import training_config


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_criterion: torch.nn.Module,
    optimizer_config: Dict,
) -> None:
    """
    Trains the model using the provided training and validation data loaders.

    Args:
        model: The model to be trained.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_criterion: The loss function to be used.
        optimizer_config: The optimizer configuration to be used.

    Returns:
        None
    """
    batch_size = training_config["batch_size"]
    num_epochs = training_config["num_epochs"]
    optimizer_name = optimizer_config["optimizer_name"]
    optimizer_params = optimizer_config["optimizer_params"]

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name} \n"
            f"Supported optimizers are: 'adam', 'sgd'."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_criterion.to(device)

    for epoch in range(num_epochs):
        epoch_loss_sum = 0.0
        window_loss_sum = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            epoch_loss_sum += loss.item()
            window_loss_sum += loss.item()

            if (i + 1) % batch_size == 0:  # print every batches
                print(
                    f"Epoch: {epoch + 1}, Batch: {i + 1}, loss: {window_loss_sum/batch_size}"
                )
                window_loss_sum = 0.0

        print(f"Epoch: {epoch + 1}, loss: {epoch_loss_sum/len(train_loader)}")

        # validation loop
        epoch_val_loss_sum = 0.0
        window_val_loss_sum = 0.0

        model.eval()

        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = loss_criterion(outputs, labels.dtype(torch.long))

                epoch_val_loss_sum += loss.item()
                window_val_loss_sum += loss.item()

                if (i + 1) % batch_size == 0:  # print every mini-batches
                    print(
                        f"Epoch: {epoch + 1}, val_Batch: {i + 1}, val_loss: {window_val_loss_sum/batch_size}"
                    )
                    window_val_loss_sum = 0.0

        print(
            f"Epoch: {epoch + 1}, val_loss: {epoch_val_loss_sum/len(val_loader)}"
        )

    print("Finished Training")
