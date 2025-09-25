from typing import Any, Dict
from transforms import DataTransform

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
    'model': 'cnn_model',
    "batch_size": 32,
    "num_epochs": 25,
    'train_loader': 'train_dataloader',
    "num_workers": 2,
}

# defining annotation file config
annotation_config: Dict[str, Any]= {
    "dataset_dir_name": "breast_cancer_data",
    "subset_name": ['train', 'test', 'valid'], 
    "destination_path": "annotation_files"
}

# defining transform config
transform_config: Dict[str, Any] = {
    "resize": (128, 128),
    "random_crop": {'size': (75, 75)},
    "random_horizontal_flip": {'p': 0.7},
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }
}

# defin target_transform config
target_transform_config = {0: 'Not detected', 1: 'Detected'}