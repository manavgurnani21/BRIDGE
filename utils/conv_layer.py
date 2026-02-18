import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch_conv_kan.kan_convs import FastKANConv1DLayer


class Conv2d(nn.Module):
    """
    A lightweight Conv2D block: Conv2d -> (optional) BatchNorm2d -> (optional) ReLU -> Dropout.

    This module is a convenience wrapper frequently used in CNN backbones. It supports
    "same" padding (for odd kernel sizes) by choosing padding = floor((k-1)/2) along
    both spatial dimensions.

    Args:
        in_channels (int):
            Number of input channels (C_in).
        out_channels (int):
            Number of output channels (C_out).
        kernel_size (tuple[int, int]):
            Kernel size (kH, kW). If `same_padding=True`, typical usage is odd kernels
            like (3, 3), (5, 5) to preserve spatial size under stride=1.
        stride (int or tuple[int, int], optional):
            Convolution stride. Default is 1.
        if_bias (bool, optional):
            Whether to include bias in the Conv2d layer. Default False.
            NOTE: If BatchNorm is enabled (bn=True), bias is often unnecessary.
        relu (bool, optional):
            If True, apply ReLU after BN. Default True.
        same_padding (bool, optional):
            If True, apply padding that approximates "same" padding:
            padding_h = floor((kH-1)/2), padding_w = floor((kW-1)/2).
            Default True.
        bn (bool, optional):
            If True, apply BatchNorm2d after convolution. Default True.

    Inputs:
        x (torch.Tensor):
            Shape (N, C_in, H, W).

    Returns:
        torch.Tensor:
            Shape (N, C_out, H_out, W_out), where spatial sizes depend on stride/padding.

    Behavior / Notes:
        - Applies dropout with p=0.3 unconditionally (controlled by model.train()/eval()).
        - If same_padding=True and kernel sizes are even, this padding is not perfectly
          symmetric "same" behavior; use odd kernels for clean size preservation.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, if_bias = False, relu=True, same_padding=True, bn=True):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True if if_bias else False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        return x


class Conv1d(nn.Module): 
    """
    A lightweight Conv1D block: Conv1d -> (optional) BatchNorm1d -> (optional) ReLU -> Dropout.

    This module is a convenience wrapper for 1D convolutional feature extractors,
    used for sequence models (e.g., RNA sequences) where the input
    shape is (N, C_in, L).

    Args:
        in_channels (int):
            Number of input channels (C_in).
        out_channels (int):
            Number of output channels (C_out).
        kernel_size (tuple[int] or int):
            Kernel size. The code assumes indexable form `kernel_size[0]`, so the most
            compatible usage is a 1-tuple like (3,) rather than integer 3.
        stride (tuple[int] or int, optional):
            Stride for convolution. Default (1,). PyTorch also accepts int.
        dilation (tuple[int] or int, optional):
            Dilation factor. Default (1,). PyTorch also accepts int.
        if_bias (bool, optional):
            Whether Conv1d has bias. Default False.
        relu (bool, optional):
            If True, apply ReLU after BN. Default True.
        same_padding (bool, optional):
            If True, uses padding = floor((k-1)/2) to approximate "same" padding when stride=1.
            Default True.
        bn (bool, optional):
            If True, apply BatchNorm1d after convolution. Default True.

    Inputs:
        x (torch.Tensor):
            Shape (N, C_in, L).

    Returns:
        torch.Tensor:
            Shape (N, C_out, L_out).

    Behavior / Notes:
        - Applies dropout p=0.3 unconditionally (active only when training).
        - For "same" length preservation, prefer odd kernel sizes with stride=1.
        - If you pass `kernel_size` as int (e.g., 3), `kernel_size[0]` will error.
          To keep this wrapper robust, call with `(3,)` or adapt the implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,),
                 dilation=(1,), if_bias=False, relu=True, same_padding=True, bn=True):
        super(Conv1d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p0,
                              dilation=dilation, bias=True if if_bias else False)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.relu = nn.SELU(inplace=True) if relu else None

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, L_out).
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        return x

    
class SimpleConvKAN_1layer(nn.Module):
    """
    A single-layer 1D convolutional block using a KAN-based convolution operator.

    This wraps `FastKANConv1DLayer` from `torch_conv_kan`, optionally followed by
    BatchNorm1d. It is intended as a drop-in replacement for standard Conv1d blocks
    when experimenting with Kolmogorov–Arnold Network (KAN) style parameterizations.

    Args:
        input_channels (int):
            Number of input channels (C_in).
        out_channels (int):
            Number of output channels (C_out).
        kernel_size (int, optional):
            Convolution kernel size. Default 3.
        groups (int, optional):
            Grouped convolution parameter passed to FastKANConv1DLayer. Default 1.
        grid_size (int, optional):
            KAN grid size controlling the resolution/complexity of the KAN function basis
            (exact meaning depends on torch_conv_kan implementation). Default 8.
        same_padding (bool, optional):
            If True, uses padding = floor((k-1)/2) to preserve length when stride=1 (odd kernels).
            Default True.
        bn (bool, optional):
            If True, apply BatchNorm1d after the KAN convolution. Default True.
        dropout (float, optional):
            Dropout probability passed into FastKANConv1DLayer (NOT applied via F.dropout here).
            Default 0.3.

    Inputs:
        x (torch.Tensor):
            Shape (N, C_in, L).

    Returns:
        torch.Tensor:
            Shape (N, C_out, L_out).

    Behavior / Notes:
        - Dropout is handled *inside* FastKANConv1DLayer via its `dropout` argument.
          The wrapper does not additionally apply F.dropout in forward (commented out).
        - `stride` and `dilation` are hard-coded to 1 in this wrapper.
    """
    def __init__(
            self,
            input_channels,
            out_channels,
            kernel_size = 3,
            groups: int = 1,
            grid_size: int = 8,
            same_padding=True,
            bn=True,
            dropout: float = 0.3):
        super(SimpleConvKAN_1layer, self).__init__()
        p0 = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = FastKANConv1DLayer(input_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=p0, stride=1, dilation=1, grid_size=grid_size, dropout=dropout)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        # self.drop = nn.Dropout(p=0.3)
        
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, L_out).
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        # x = F.dropout(x, 0.3, training=self.training)
        return x
