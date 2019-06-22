import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, size, num_classes, device):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.size = size
        self.num_classes = num_classes
        self.device = device

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(
            Image.open(self.image_paths[idx]).resize((self.size, self.size),
                                                     resample=Image.LANCZOS))
        # Normalize between 0 and 1
        image = image / 255
        # Standardize to pretrained input image values
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  #
        mask = np.array(
            Image.open(self.mask_paths[idx]).resize((self.size, self.size),
                                                    resample=Image.NEAREST),
            dtype='int')[:, :, 0]
        image = np.moveaxis(image, -1, 0)
        image = torch.from_numpy(image).float()
        image = image.to(self.device)
        mask = np.moveaxis(mask, -1, 0)
        mask = torch.from_numpy(mask).long()
        mask = mask.to(self.device)
        return image, mask

# def one_hot_encode(mask, num_classes):
#     y = mask.ravel()
#     one_hot = np.zeros((y.shape[0], num_classes))
#     one_hot[np.arange(y.shape[0]), y] = 1
#     return np.reshape(one_hot, mask.shape + (num_classes,))
