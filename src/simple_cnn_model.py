from typing import Dict

import torch
import torch.nn as nn

# import torch.functional as F


class ImageClassifierCNN(nn.Module):
    """
    A Convolutional Neural Network for image classification built from a config.
    """

    def __init__(
        self,
        input_channel: int,
        num_classes: int,
        conv_config: list,
        fc_config: list,
        input_height: int = 64,
        input_width: int = 64,
    ) -> None:
        """
        Initializes the CNN model based on the provided configuration.
        Args:
            input_channel (int): Number of input channels (e.g., 3 for RGB images).
            n_out_class (int): Number of output classes for classification.
            conv_config (Dict): Configuration for convolutional layers.
            fc_config (Dict): Configuration for fully connected layers.

        Example conv_config:
        [
            {"out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
            {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}
        ]
        Example fc_config:
        [
            {"out_dim": 128, "dropout": 0.5},
            {"out_dim": 64}
        ]
        """
        super().__init__()

        conv_layers: list[nn.Module] = []
        conv_in_channel = input_channel

        for layer in conv_config:
            out_channel = layer.get("out_channels")
            conv_layers.append(nn.Conv2d(in_channels=conv_in_channel, **layer))
            conv_layers.append(nn.BatchNorm2d(num_features=out_channel))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2))

            conv_in_channel = out_channel

        self.conv = nn.Sequential(*conv_layers)

        # flatten output layer for fc layers
        dummy = torch.rand(1, input_channel, input_height, input_width)
        flat_dim = self._flatten_layer(dummy)

        fc_layers: list[nn.Module] = []
        in_dim = flat_dim

        for hidden_layer in fc_config:
            out_dim = hidden_layer["out_dim"]
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.ReLU())

            if "dropout" in hidden_layer:
                fc_layers.append(nn.Dropout(p=hidden_layer["dropout"]))

            in_dim = out_dim

        fc_layers.append(
            nn.Linear(in_features=in_dim, out_features=num_classes)
        )

        self.fc = nn.Sequential(*fc_layers)

    def _flatten_layer(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            y = self.conv(x)
        return torch.flatten(y, 1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    conv_layers = [
        {
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        },
        {
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": "same",
        },
    ]

    fc_layers = [{"out_dim": 64, "dropout": 0.5}, {"out_dim": 32}]

    model = ImageClassifierCNN(
        input_channel=3,
        num_classes=10,
        conv_config=conv_layers,
        fc_config=fc_layers,
    )

    print(model)

    # Test with a dummy input
    dummy_input = torch.randn(32, 3, 64, 64)
    output = model(dummy_input)

    print("Output shape:", output.shape)
