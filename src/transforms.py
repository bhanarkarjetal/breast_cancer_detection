import torch
from torchvision.transforms import v2

class data_transform():
    def __init__ (self, resize = None, 
                   random_crop = None,
                   normalize = None,
                   random_horizontal_flip = None,
                   random_vertical_flip = None,
                   random_photometric_distort = None
                  ):
        self.resize = resize
        self.random_crop = random_crop
        self.normalize = normalize
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.random_photometric_distort = random_photometric_distort

    def transform(self):
        transform_list = []
        if self.resize:
            transform_list.append(v2.Resize(self.resize))
        else:
            transform_list.appen(v2.Resize(64,64))

        try:
            if self.random_crop:
                transform_list.append(v2.RandomCrop(self.random_crop))
        except Exception:
            print('Crop size greater than resize size')

        if self.random_horizontal_flip:
            transform_list.append(v2.RandomHorizontalFlip())

        if self.random_vertical_flip:
            transform_list.append(v2.RandomVerticalFlip())

        if self.random_photometric_distort:
            transform_list.append(v2.RandomPhotometricDistort())

        transform_list.append(v2.ToImage())
        transform_list.append(v2.ToDtype(torch.float, scale = True))

        if self.normalize:
            transform_list.append(v2.Normalize())
        else:
            transform_list.append(v2.Normalize(mean=[0.485, 0.456, 0.406], 
                                               std=[0.229, 0.224, 0.225]))
        

        
        return v2.Compose(transform_list)