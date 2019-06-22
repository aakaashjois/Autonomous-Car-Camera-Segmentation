import numpy as np
import torch
from torch.nn import Module
from torch.nn.functional import cross_entropy, softmax


class CustomLoss(Module):
    def __init__(self, num_classes, device):
        super(CustomLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

    def get_tversky_loss(self, pred, target):
        target_oh = torch.eye(self.num_classes)[target.squeeze(1)]
        target_oh = target_oh.permute(0, 3, 1, 2).float()
        probs = softmax(pred, dim=1)
        target_oh = target_oh.type(pred.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        inter = torch.sum(probs * target_oh, dims)
        fps = torch.sum(probs * (1 - target_oh), dims)
        fns = torch.sum((1 - probs) * target_oh, dims)
        t = (inter / (inter + 0.5 * (fps + fns))).mean()
        return 1 - t

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
        ce = cross_entropy(pred, target,
                           weight=self.get_dice_coef(pred, target).to(
                               self.device))
        tv = self.get_tversky_loss(pred, target)
        loss = (0.5 * ce) + (0.5 * tv)
        return loss
