import numpy as np
import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy


class CustomLoss(Module):
    def __init__(self, num_classes, device):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def get_dice_coef(self, pred, target):
        pred_class = torch.argmax(pred, dim=1)
        d_coef = np.ones(self.num_classes)
        for c in range(self.num_classes):
            p = (pred_class == c)
            t = (target == c)
            inter = (p * t).sum().float()
            union = p.sum() + t.sum() + 1e-6
            d = 2 * inter / union
            d_coef[c] = 1 - d
        return torch.from_numpy(d_coef).float()

    def forward(self, pred, target):
        loss = cross_entropy(pred, target,
                             weight=self.get_dice_coef(pred, target).to(
                                 self.device))
        return loss
