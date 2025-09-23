import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BreastCancerDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        
        if idx > len(self):
            raise IndexError("Given index is beyond the length of dataset.")
        
        img_name = os.path.join(self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1]

        return image, label
