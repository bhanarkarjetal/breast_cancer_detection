from torch.utils.data import DataLoader
from create_dataset import BreastCancerDataset
from typing import Optional, Callable

class DataLoaderWrapper(DataLoader):
    def __init__(self, annotation_file: str, 
                 transforms: Optional[Callable] = None, 
                 root_dir: str = '',
                 batch_size: int = None, 
                 shuffle: bool = True):
        
        self.annotation_file = annotation_file
        self.transforms = transforms
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load_data(self):
        dataset = BreastCancerDataset(csv_file = self.annotation_file,
                                     root_dir = self.root_dir,
                                     transform = self.transforms)
        
        data_loader = DataLoader(dataset, 
                                 batch_size = self.batch_size, 
                                 shuffle = self.shuffle)
        
        return data_loader