from typing import Any, Dict, Optional

import torch
from torchvision.transforms import v2


class DataTransform:
    def __init__(self, user_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the DataTransform class with user-defined or default configurations.
        Args:
            user_config (Optional[Dict[str, Any]]): User-defined transformations and their parameters.
        """

        # Define default transformations and their parameters
        self.default_config: Dict[str, Any] = {
            "resize": {"size": (128, 128)},
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        }

        # Override defaults with user configurations if provided
        self.config = {**self.default_config, **(user_config or {})}

    def get_transform(self):
        """
        Build and returns a torchvision.transforms.Compose object based on the config.

        Returns:
            torchvision.transforms.Compose: Composed transformations.
        """

        transform_list = []

        # Map string names to torchvision transforms
        transform_map = {
            "resize": v2.Resize,
            "random_horizontal_flip": v2.RandomHorizontalFlip,
            "random_vertical_flip": v2.RandomVerticalFlip,
            "random_rotation": v2.RandomRotation,
            "brightness_contrast": v2.ColorJitter,
            "random_crop": v2.RandomCrop,
            "normalize": v2.Normalize,
        }

        for key, params in self.config.items():
            if key == 'normalize':
                 normalize_config = params
            
            elif key in transform_map:
                transform_list.append(transform_map[key](**params))

            else:
                raise ValueError(f"Transform '{key}' is not recognized.")

        transform_list.append(v2.ToImage())
        transform_list.append(v2.ToDtype(torch.float, scale=True))
        transform_list.append(v2.Normalize(**normalize_config))

        return v2.Compose(transform_list)
