import torch
from torchvision.transforms import v2
from typing import Optional, Tuple, List, Dict

class DataTransform():
    def __init__ (self, resize: Optional[Tuple[int, int]] = None, 
                   random_crop: Optional[Tuple[int, int]] = None,
                   normalize: Optional[Dict[str, List[float]]] = None,
                   random_horizontal_flip: Optional[float] = None,
                   random_vertical_flip: Optional[float] = None,
                   random_photometric_distort: Optional[Dict[str, List[float]]] = None,
                   default_resize = (64, 64),
                   default_normalize_mean = [0.485, 0.456, 0.406],
                   default_normalize_std = [0.229, 0.224, 0.225]):
        
        self.resize = resize
        self.random_crop = random_crop
        self.normalize = normalize
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.random_photometric_distort = random_photometric_distort
        self.default_resize = default_resize
        self.default_normalize_mean = default_normalize_mean
        self.default_normalize_std = default_normalize_std

    def transform(self):
        transform_list = []
        if self.resize is not None:
            transform_list.append(v2.Resize(self.resize))
        else:
            transform_list.append(v2.Resize((self.default_resize)))

        if self.random_crop is not None:
            transform_list.append(v2.RandomCrop(self.random_crop))

        if self.random_horizontal_flip is True:
            transform_list.append(v2.RandomHorizontalFlip(self.random_horizontal_flip))

        if self.random_vertical_flip is True:
            transform_list.append(v2.RandomVerticalFlip(self.random_vertical_flip))

        if self.random_photometric_distort is True:
            transform_list.append(v2.RandomPhotometricDistort(self.random_photometric_distort))

        transform_list.append(v2.ToImage())
        transform_list.append(v2.ToDtype(torch.float, scale = True))

        if self.normalize is not None:
            transform_list.append(v2.Normalize(self.normalize['mean'], 
                                               self.normalize['std']))
        else:
            transform_list.append(v2.Normalize(mean=self.default_normalize_mean, 
                                               std=self.default_normalize_std))
        
        return v2.Compose(transform_list)