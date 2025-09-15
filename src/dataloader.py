from torch.utils.data import DataLoader
from create_dataset import BreastCancerDataset
from typing import Optional, Callable

def get_data_loader(
    annotation_file: str,
    root_dir: str,
    transforms: Optional[Callable] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2
):
    """
    Utility function to create a DataLoader for the BreastCancerDataset.
    
    Args:
        annotation_file (str): Path to the CSV file with annotations.
        root_dir (str): Directory with all the images.
        transforms (callable, optional): Optional transformations to be applied on a sample.
        batch_size (int, optional): Number of samples per batch to load. Default is 32.
        shuffle (bool, optional): Set to True to have the data reshuffled at every epoch. Default is True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is 2.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """

    dataset = BreastCancerDataset(
        csv_file=annotation_file,
        root_dir=root_dir,
        transform=transforms
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader