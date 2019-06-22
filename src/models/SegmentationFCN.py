from torch import cat
from torch import nn
from torchvision import models


class SegmentationFCN(nn.Module):

    def __init__(self, n_class, device):
        super().__init__()
        self.device = device
        self.backend = models.resnet18(pretrained=True)
        layers = list(self.backend.children())
        self.layer1 = nn.Sequential(*layers[:5])
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear',
                                     align_corners=False)
        self.layer2 = layers[5]
        self.upsample2 = nn.Upsample(scale_factor=8, mode='bilinear',
                                     align_corners=False)
        self.layer3 = layers[6]
        self.upsample3 = nn.Upsample(scale_factor=16, mode='bilinear',
                                     align_corners=False)
        self.layer4 = layers[7]
        self.upsample4 = nn.Upsample(scale_factor=32, mode='bilinear',
                                     align_corners=False)

        self.conv = nn.Conv2d(64 + 128 + 256 + 512, n_class, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(self.device)
        x = self.layer1(x)
        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)

        merge = cat([up1, up2, up3, up4], dim=1)
        merge = self.conv(merge)
        out = self.softmax(merge)
        return out.float()
