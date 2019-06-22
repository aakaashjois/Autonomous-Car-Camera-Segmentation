from torch.nn import Module, ModuleList, Conv2d, MaxPool2d, ConvTranspose2d, \
    BatchNorm2d, Sequential
from torch.nn import functional as f
from torch import cat


def conv(in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
    c = Conv2d(in_channels, out_channels, kernel_size=kernel_size,
               stride=1, padding=padding)
    if batch_norm:
        bn = BatchNorm2d(out_channels)
        return Sequential(c, bn)
    else:
        return c


class DownConv(Module):
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv(self.in_channels, self.out_channels)
        self.conv2 = conv(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.upconv = ConvTranspose2d(self.in_channels, self.out_channels,
                                      kernel_size=2, stride=2)

        self.conv1 = conv(2 * self.out_channels, self.out_channels)
        self.conv2 = conv(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up, output_size=from_down.size())
        x = cat((from_up, from_down), 1)
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        return x


class SegmentationUNet(Module):
    def __init__(self, num_classes, device, in_channels=3, depth=7,
                 start_filts=64):
        super(SegmentationUNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.device = device

        self.down_convs = []
        self.up_convs = []

        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs)
            self.up_convs.append(up_conv)

        self.conv_final = conv(outs, self.num_classes, kernel_size=1, padding=0,
                               batch_norm=False)

        self.down_convs = ModuleList(self.down_convs)
        self.up_convs = ModuleList(self.up_convs)

    def forward(self, x):
        x = x.to(self.device)
        encoder_outs = []

        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        return x
