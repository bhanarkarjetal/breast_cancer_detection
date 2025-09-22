from typing import Any, Dict

# defining model_config for simple_cnn_model.py
model_config: Dict[str, Any] = {
    "input_channels": 3,
    "num_classes": 2,
    "conv_config": [
        {"out_channels": 16, "kernel_size": 3, "padding": 1},
        {"out_channels": 32, "kernel_size": 3, "padding": 1},
        {"out_channels": 64, "kernel_size": 3, "padding": 1},
    ],
    "fc_config": [
        {"out_features": 256, "dropout": 0.5},
        {"out_features": 128, "dropout": 0.5},
    ],
    "input_height": 64,
    "input_width": 64,
}

# defining optimizer_config for optimizer.py
optimizer_config: Dict[str, Any] = {
    "optimizer_name": "adam",
    "optimizer_params": {"lr": 0.001},
}

# defining training_config for train.py
training_config: Dict[str, int] = {
    "batch_size": 32,
    "num_epochs": 25,
    "num_workers": 4,
}
