from torch.utils.data import DataLoader, Dataset



def get_data_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """
    Utility function to create a DataLoader for the BreastCancerDataset.

    Args:
        dataset (Dataset): Dataset instance on which the DataLoader to be applied
        batch_size (int, optional): Number of samples per batch to load. Default is 32.
        shuffle (bool, optional): Set to True to have the data reshuffled at every epoch. Default is True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is 2.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader
