from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def auto_pad(k: int, s: int = 1, d: int = 1):
    """
    Automatically padding.
    output = floor((input + 2 * p - k) / s) + 1
    :param k: kernel size
    :param s: stride
    :param d: dilation
    """
    k_effe = d * (k - 1) + 1  # effective kernel size, considering dilation
    if k_effe % 2 == 1:  # odd kernel size
        return (k_effe - s + 1) // 2
    else:  # even kernel size
        return (k_effe - s) // 2


class Linear(nn.Module):
    """
    Linear / Fully-connected layer. No batch norm or layer norm.
    """

    def __init__(self, c_in: int, c_out: int, dropout=0.1, act: bool=True, bias: bool=False):
        super().__init__()
        self.act = ReLU() if act else nn.Identity()
        self.layers = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(c_in, c_out, bias=bias))

    def forward(self, x):
        x = self.layers(x)
        x = self.act(x)
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron, MLP, feed-forward network, FFN.
    """

    def __init__(self, c_in, c_hidden, c_output, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [c_hidden] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([c_in] + h, h + [c_output]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Conv(nn.Module):
    """2D convolution layer, without batch norm."""

    def __init__(self, c_in, c_out, k=1, s=1, act: bool=True, bias: bool=False):
        """
        Initializes the 2D convolution layer.
        :param c_in: input channels
        :param c_out: output channels
        :param k: kernel size
        :param s: stride
        """
        super().__init__()
        p = auto_pad(k, s)
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=bias)
        self.act = ReLU() if act else nn.Identity()  # TODO: optimize the activation func.

    def forward(self, x):
        return self.act(self.conv(x))


class ConvBN(nn.Module):
    """2D convolution layer, with batch norm."""

    def __init__(self, c_in, c_out, k=1, s=1, act: bool=True, bias: bool=False):
        super().__init__()
        p = auto_pad(k, s)
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = ReLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BottleneckBlock(nn.Module):
    """Bottleneck block in ResNet."""

    def __init__(self, c_in, c_hidden, s, expansion=4):
        """
        Initializes the Bottleneck block.
        :param c_in: input channels
        :param c_hidden: hidden channels
        :param s: stride for the 2nd conv layer (could be s=2 for feature map downsample)
        :param expansion: expansion factor of the output channels to the hidden channels
        """
        super().__init__()
        c_out = c_hidden * expansion
        self.conv_1 = ConvBN(c_in, c_hidden, k=1, s=1)
        self.conv_2 = ConvBN(c_hidden, c_hidden, k=3, s=s)
        self.conv_3 = ConvBN(c_hidden, c_out, k=1, s=1, act=False)
        self.act = ReLU()

        # Shortcut connection / skip connection
        if c_in == c_out and s == 1:  # input and output with same spatial size and feature dim
            self.shortcut = nn.Identity()  # no BN for identity
        else:
            self.shortcut = ConvBN(c_in, c_out, k=1, s=s, act=False)

    def forward(self, x):
        x_id = x  # identity of x, for shortcut connection
        x_out = self.conv_1(x)  # conv, bn, relu
        x_out = self.conv_2(x_out)  # conv, bn, relu
        x_out = self.conv_3(x_out)  # conv, bn
        x_shortcut = self.shortcut(x_id)  # conv, bn / identity
        x_out = self.act(x_out + x_shortcut)  # add, relu
        return x_out


class BuildingBlock(nn.Module):
    """Building block in ResNet."""

    def __init__(self, c_in, c_hidden, s, expansion=1):
        super().__init__()
        # Do not use the expansion parameters.
        c_out = c_hidden
        self.conv_1 = ConvBN(c_in, c_hidden, k=3, s=s)
        self.conv_2 = ConvBN(c_hidden, c_out, k=3, s=1, act=False)
        self.act = ReLU()

        # Shortcut connection / skip connection.
        if c_in == c_out and s == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = ConvBN(c_in, c_out, k=1, s=s, act=False)

    def forward(self, x):
        x_id = x
        x_out = self.conv_1(x)
        x_out = self.conv_2(x_out)
        x_shortcut = self.shortcut(x_id)
        x_out = self.act(x_out + x_shortcut)
        return x_out


class ResNetLayer(nn.Module):
    """A standard ResNet layer with n Bottleneck blocks."""

    def __init__(self, c_in, c_hidden, n_blocks, bottleneck=True, downsample=True, expansion=4):
        """
        Initializes the ResNet layer.
        :param c_in: input channels
        :param c_hidden: hidden channels
        :param n_blocks: number of bottleneck blocks in resnet layer
        :param bottleneck: whether to use bottleneck block or basic building block
        :param expansion: expansion factor of the output channels to the hidden channels
        :param downsample: whether to downsample the input feature map in the first block
        """
        super().__init__()
        c_out = c_hidden * expansion

        # Handle the first block, could be a downsample block.
        resnet_layer = nn.ModuleList()
        Block = BottleneckBlock if bottleneck else BuildingBlock
        if downsample:
            resnet_layer.append(Block(c_in, c_hidden, s=2, expansion=expansion))
        else:
            resnet_layer.append(Block(c_in, c_hidden, s=1, expansion=expansion))

        # Handle the rest (n_blocks - 1)
        resnet_layer.extend([Block(c_out, c_hidden, s=1, expansion=expansion) for _ in range(n_blocks - 1)])
        self.resnet_layer = nn.Sequential(*resnet_layer)

    def forward(self, x):
        return self.resnet_layer(x)


class VggBlock(nn.Module):
    """Standard VGG block."""

    def __init__(self, c_in, c_out, n_layers, k=3, s=1):
        """
        Initializes the VGG block.
        :param c_in: input channels
        :param c_out: output channels
        :param n_layers: number of conv layers in a Vgg block
        :param k: kernel size
        :param s: stride
        """
        super().__init__()
        # Handle the first layer
        vgg_block = nn.ModuleList([Conv(c_in, c_out, k=k, s=s, bias=True)])

        # Handle the rest layers
        vgg_block.extend([Conv(c_out, c_out, k=k, s=s, bias=True) for _ in range(n_layers - 1)])
        self.vgg_block = nn.Sequential(*vgg_block)

    def forward(self, x):
        return self.vgg_block(x)


class AvgPool(nn.Module):
    """Average pooling layer, global pooling or given kernel and stride."""

    def __init__(self, glb: bool=True, k=None, s=None):
        """
        Initialize the average pooling layer.
        :param glb: global pooling or local pooling
        :param k: kernel size (optional, only when glb is False)
        :param s: stride (optional, only when glb is False)
        """
        super().__init__()
        if glb:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            p = auto_pad(k, s)
            self.pool = nn.AvgPool2d(kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return self.pool(x)


class MaxPool(nn.Module):
    """Max pooling layer, global pooling or given kernel and stride."""

    def __init__(self, glb: bool=True, k=None, s=None):
        """
        Initialize the max pooling layer.
        :param glb: global pooling or local pooling
        :param k: kernel size (optional, only when glb is False)
        :param s: stride (optional, only when glb is False)
        """
        super().__init__()
        if glb:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            p = auto_pad(k, s)
            self.pool = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return self.pool(x)


class Upsample(nn.Module):
    """Up-sampling layer by interpolation."""

    def __init__(self, scale, mode='bilinear'):
        """
        Initializes the up-sampling layer.
        :param scale: up-sampling scale factor
        :param mode: "nearest", "bilinear"
        """
        super().__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode)


class Flatten(nn.Module):

    def __init__(self, dim_s=1, dim_e=-1):
        super().__init__()
        self.flatten = nn.Flatten(dim_s, dim_e)

    def forward(self, x):
        return self.flatten(x)


class Concat(nn.Module):
    """Concatenates a list of tensors along a given dimension."""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.dim)


class ReLU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return self.relu(x)


class SiLU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.silu = nn.SiLU(inplace=inplace)
    def forward(self, x):
        return self.silu(x)


class LeakyReLU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.leaky = nn.LeakyReLU(inplace=inplace)
    def forward(self, x):
        return self.leaky(x)