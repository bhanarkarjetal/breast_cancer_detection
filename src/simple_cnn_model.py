from typing import Dict, Any
import torch
import torch.nn as nn


class ImageClassifierCNN(nn.Module):
    """
    A Convolutional Neural Network for image classification built from a config.
    """

    def __init__(
        self,
        input_channel: int,
        num_classes: int,
        conv_config: list[Dict],
        fc_config: list[Dict],
    ) -> None:
        """
        Initializes the CNN model with conv layers.
        FC layers are built lazily when the first input is passed.
        """
        super().__init__()
        self.num_classes = num_classes
        self.fc_config = fc_config
        self.flatten_dim = None  # computed at runtime

        # Build convolutional layers
        conv_layers: list[nn.Module] = []
        conv_in_channel = input_channel

        for layer in conv_config:
            out_channel = layer.pop("out_channels")
            conv_kwargs = {
                k: v
                for k, v in layer.items()
                if k in ["kernel_size", "stride", "padding"]
            }

            conv_layers.append(
                nn.Conv2d(in_channels=conv_in_channel, out_channels=out_channel, **conv_kwargs)
            )
            conv_layers.append(nn.BatchNorm2d(num_features=out_channel))
            conv_layers.append(nn.ReLU(inplace=True))

            if "dropout" in layer:
                conv_layers.append(nn.Dropout2d(layer["dropout"]))

            if layer.get("pool", True):
                conv_layers.append(nn.MaxPool2d(kernel_size=2))

            conv_in_channel = out_channel

        self.conv = nn.Sequential(*conv_layers)
        self.fc = None  # will be created after flatten_dim is known

    def _build_fc(self, flat_dim: int) -> None:
        """
        Build FC layers dynamically based on computed flat_dim.
        """
        fc_layers: list[nn.Module] = []
        in_dim = flat_dim

        for hidden_layer in self.fc_config:
            out_dim = hidden_layer["out_features"]
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.ReLU(inplace=True))

            if "dropout" in hidden_layer:
                fc_layers.append(nn.Dropout(p=hidden_layer["dropout"]))

            in_dim = out_dim

        fc_layers.append(nn.Linear(in_features=in_dim, out_features=self.num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)

        # Build FC on first forward pass
        if self.fc is None:
            self.flatten_dim = x.size(1)
            self._build_fc(self.flatten_dim)
            # move fc to same device as conv
            self.fc.to(x.device)

        x = self.fc(x)
        return x


if __name__ == "__main__":
    conv_layers = [
        {
            "out_channels": 16,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
        {
            "out_channels": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
        },
    ]

    fc_layers = [{"out_features": 64, "dropout": 0.5}, {"out_features": 32}]

    model = ImageClassifierCNN(
        input_channel=3,
        num_classes=1,
        conv_config=conv_layers,
        fc_config=fc_layers
    )

    # Test with a dummy input
    dummy_input = torch.randn(32, 3, 64, 64)
    output = model(dummy_input)

    print("Output shape:", output.shape)
