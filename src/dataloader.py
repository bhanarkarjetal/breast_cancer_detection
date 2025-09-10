import torch
from torch.utils.data import DataLoader
from create_dataset import BreastCancerDataset

class dataloader(DataLoader):
    def __init__(self, annotation_file, 
                 transforms, 
                 root_dir,
                 batch_size = None, 
                 shuffle = True):
        
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