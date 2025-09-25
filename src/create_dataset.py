import os
from typing import  Optional, Callable, Tuple, Union
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class BreastCancerDataset(Dataset):
    def __init__(self, 
                 csv_file: str, 
                 image_transform: Optional[Callable] = None):
        
        """ Build custom dataset for brest cancer images.
        
        Args: 
            csv_file (str): path to csv file with 'image_path' and ' label' columns
            image_transform (Callable, Optional): Transformations to applu to images
        """

        self.data_frame = pd.read_csv(csv_file).reset_index(drop = True)
        self.image_transform = image_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, int]:
        
        # Handle integer index bounds checking

        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Get the row using pandas iloc
        row = self.data_frame.iloc[idx]
        img_path = row['image_path']
        label = int(row['label'])  # Ensure label is an integer

        # Check if image exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')

        if self.image_transform:
            image = self.image_transform(image)

        return image, int(label)