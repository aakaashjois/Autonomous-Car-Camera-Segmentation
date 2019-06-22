import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader

from SegmentationDataset import SegmentationDataset
from SegmentationUNet import SegmentationUNet
from TverskyCrossEntropyDiceWeightedLoss import \
    TverskyCrossEntropyDiceWeightedLoss


class SegmentationAgent:
    def __init__(self, val_percentage, test_num, num_classes,
                 batch_size, img_size, data_path, shuffle_data,
                 learning_rate, device):
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.img_size = img_size
        self.images_list, self.masks_list = self.load_data(data_path)
        train_split, val_split, test_split = self.make_splits(
                val_percentage, test_num, shuffle_data)
        self.train_loader = self.get_dataloader(train_split)
        self.validation_loader = self.get_dataloader(val_split)
        self.test_loader = self.get_dataloader(test_split)
        self.model = SegmentationUNet(self.num_classes, self.device)
        self.criterion = TverskyCrossEntropyDiceWeightedLoss(self.num_classes,
                                                             self.device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def load_data(self, path):
        images_list = list(path.glob('*/*/CameraRGB/*.png'))
        masks_list = list(path.glob('*/*/CameraSeg/*.png'))
        if len(images_list) != len(masks_list):
            raise ValueError('Invalid data loaded')
        images_list = np.array(images_list)
        masks_list = np.array(masks_list)
        return images_list, masks_list

    def make_splits(self, val_percentage, test_num, shuffle=True):
        if shuffle:
            shuffle_idx = np.random.permutation(range(len(self.images_list)))
            self.images_list = self.images_list[shuffle_idx]
            self.masks_list = self.masks_list[shuffle_idx]

        val_num = len(self.images_list) - int(val_percentage * len(
                self.images_list))
        train_images = self.images_list[:val_num]
        train_masks = self.masks_list[:val_num]

        validation_images = self.images_list[val_num:-test_num]
        validation_masks = self.masks_list[val_num:-test_num]

        test_images = self.images_list[-test_num:]
        test_masks = self.masks_list[-test_num:]

        return (train_images, train_masks), \
               (validation_images, validation_masks), \
               (test_images, test_masks)

    def get_dataloader(self, split):
        return DataLoader(SegmentationDataset(split[0], split[1], self.img_size,
                                              self.num_classes, self.device),
                          self.batch_size, shuffle=True)
