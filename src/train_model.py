from typing import Dict

import torch

def train_model(
    model: torch.nn.Module,
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    Trains the model using the provided training and validation data loaders.

    Args:
        model: The model to be trained.
        num_epochs: Number of training epochs
        batch_size: mini batch size
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_criterion: The loss function to be used.
        optimizer_config: The optimizer configuration to be used.

    Returns:
        Dict with training/ validation loss history
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_criterion.to(device)
    
    for epoch in range(num_epochs):
        # training loop
        model.train()
        epoch_loss_sum = 0.0

        # for idx, data in enumerate(train_loader, start=1):
        for inputs, labels in train_loader:
            # inputs, labels = data
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            epoch_loss_sum += loss.item() * inputs.size(0)


        avg_train_loss = epoch_loss_sum/len(train_loader.dataset)

        # validation loop
        if val_loader is not None:
            epoch_val_loss_sum = 0.0

            model.eval()

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)

                    outputs = model(inputs)
                    loss = loss_criterion(outputs, labels) 

                    epoch_val_loss_sum += loss.item() * inputs.size(0)

            avg_val_loss = epoch_val_loss_sum/len(val_loader.dataset)

            print(
                f"Epoch: {epoch + 1} | train_loss_avg: {avg_train_loss:.4f} | val_loss_avg: {avg_val_loss:.4f}")
        else:
            print(
                f"Epoch: {epoch + 1} | train_loss_avg: {avg_train_loss:.4f}")

    print("Finished Training")
