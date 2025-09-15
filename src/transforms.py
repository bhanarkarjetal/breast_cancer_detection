import torch
from torchvision.transforms import v2
from typing import Any, Optional, Tuple, List, Dict

class DataTransform():
    def __init__ (self, user_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the DataTransform class with user-defined or default configurations.
        Args:
            user_config (Optional[Dict[str, Any]]): User-defined transformations and their parameters.
        """

        # Define default transformations and their parameters
        self.default_config = {
            'resize': {'size': (256, 256)},
            'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'random_horizontal_flip': {'p': 0.5}
        }

        # Override defaults with user configurations if provided
        self.config = self.default_config.copy()
        if user_config:
            self.config.update(user_config)


    def transform(self):
        '''
        Build and returns a torchvision.transforms.Compose object based on the config.
        
        Returns:
            torchvision.transforms.Compose: Composed transformations.
        '''
        
        transform_list = []

        # Map string names to torchvision transforms
        transform_map = {
            'resize': v2.Resize,
            'random_crop': v2.RandomCrop,
            'random_horizontal_flip': v2.RandomHorizontalFlip,
            'random_vertical_flip': v2.RandomVerticalFlip,
            'random_photometric_distort': v2.RandomPhotometricDistort,
            'to_image': v2.ToImage,
            'to_dtype': v2.ToDtype,
            'normalize': v2.Normalize
        }

        for key, transform in self.user_config.items():
            if key in transform_map:
                transform_list.append(transform_map[key](**transform))
            else:
                raise ValueError(f"Transform '{key}' is not recognized.")
            
        transform_list.append(v2.ToImage())
        transform_list.append(v2.ToDtype(torch.float, scale = True))

        return v2.Compose(transform_list)