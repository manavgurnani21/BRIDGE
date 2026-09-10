"""
(Deprecated) Residual and Squeeze-and-Excitation blocks kept for backward compatibility.

This module defines a small set of PyTorch building blocks that were used in earlier
experiments (or in alternative backbones) but are **not** used by the current BRIDGE
implementation.

.. warning::
   Deprecated / unused in BRIDGE

   The current BRIDGE model does not import or rely on any classes in this file.
   This module is kept only to support older checkpoints/scripts and to preserve
   reproducibility of legacy experiments. New code should not depend on it.

What is inside
--------------
Squeeze-and-Excitation (SE) gates
    ``SEBlock``:
        1D SE gate operating on tensors shaped ``(B, C, L)`` using ``AdaptiveAvgPool1d(1)``.
        Returns the channel gate shaped ``(B, C, 1)`` (note: this implementation returns the
        gate itself, not ``x * gate``; multiplication is expected to be done by the caller).

    ``SEBlock_``:
        2D SE gate operating on tensors shaped ``(B, C, H, W)`` using ``AdaptiveAvgPool2d(1)``.
        Returns the channel gate shaped ``(B, C, 1, 1)``.

Residual bottleneck-style blocks
    ``ResidualBlock1D`` and ``ResidualBlock2D``:
        Standard residual blocks with three convolutions:
        ``1x1 -> kxk -> 1x1`` plus BatchNorm and ReLU. Optional projection shortcut
        (``downsample=True``) to match shapes.

    ``ResidualBlock1D_`` and ``ResidualBlock2D_``:
        Larger-expansion variants (different channel expansion factors). These were used in
        some earlier architectures and may not match the current project's channel layout.

Input / output conventions
--------------------------
All blocks are channel-first:

- 1D blocks expect ``x`` shaped ``(B, C, L)``.
- 2D blocks expect ``x`` shaped ``(B, C, H, W)``.

The residual blocks preserve spatial dimensions when stride=1 and padding is chosen
appropriately (as implemented). Channel dimensions may change due to expansion, in which
case the projection shortcut is applied.

How to use (legacy only)
------------------------
Example: apply a residual block to a 1D feature map:

.. code-block:: python

    import torch
    from legacy_blocks import ResidualBlock1D

    x = torch.randn(8, 64, 101)  # (B, C, L)
    block = ResidualBlock1D(planes=64, downsample=True)
    y = block(x)

Example: SE gate (note it returns the gate):

.. code-block:: python

    from legacy_blocks import SEBlock

    x = torch.randn(8, 64, 101)
    gate = SEBlock(channel=64)(x)      # (8, 64, 1)
    y = x * gate                        # caller multiplies

Notes and caveats
-----------------
- Naming:
  Classes with a trailing underscore (``*_``) are alternative variants and are not
  guaranteed to be API-stable.
- SE behavior:
  ``SEBlock`` / ``SEBlock_`` return only the gating tensor (sigmoid outputs), not the
  gated activations. This is easy to misread if you expect "SE block" to return ``x * gate``.

"""

import torch  # core tensor library (not directly used beyond nn/F below, kept as in original)
import torch.nn as nn  # layer/module base classes
import torch.nn.functional as F  # functional ops (unused in this file, kept as in original)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()  # standard nn.Module init
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # collapse the length axis to a single value per channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * reduction),  # expand channel dim (bottleneck-style gating MLP, widening here)
            nn.ReLU(inplace=True),  # nonlinearity
            nn.Linear(channel * reduction, channel),  # project back to the original channel count
            nn.Sigmoid()  # squash to (0, 1) per-channel gate values
        )

    def forward(self, x):
        b, c, _ = x.size()  # batch size and channel count (length dim discarded)
        y = self.avg_pool(x).view(b, c)  # (B, C, L) -> (B, C, 1) -> (B, C) global average per channel
        y = self.fc(y).view(b, c, 1)  # compute the gate and reshape to broadcast over the length axis
        return y  # (B, C, 1) channel gate; caller is expected to multiply this into x


class SEBlock_(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock_, self).__init__()  # standard nn.Module init
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # collapse spatial (H, W) axes to a single value per channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # bottleneck: reduce channel dim
            nn.ReLU(inplace=True),  # nonlinearity
            nn.Linear(channel // reduction, channel),  # project back to the original channel count
            nn.Sigmoid()  # squash to (0, 1) per-channel gate values
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # batch size and channel count (spatial dims discarded)
        y = self.avg_pool(x).view(b, c)  # (B, C, H, W) -> (B, C, 1, 1) -> (B, C) global average per channel
        y = self.fc(y).view(b, c, 1, 1)  # compute the gate and reshape to broadcast over spatial dims
        return y  # (B, C, 1, 1) channel gate; caller is expected to multiply this into x


class ResidualBlock2D_(nn.Module):

    def __init__(self, planes, kernel_size=(11, 5), padding=(5, 2), downsample=True):
        super(ResidualBlock2D_, self).__init__()  # standard nn.Module init
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)  # 1x1 conv, channels unchanged
        self.b1 = nn.BatchNorm2d(planes)  # normalize after c1
        self.c2 = nn.Conv2d(planes, planes * 2, kernel_size=kernel_size, stride=1,
                            padding=padding, bias=False)  # kxk conv, expands channels by 2x
        self.b2 = nn.BatchNorm2d(planes * 2)  # normalize after c2
        self.c3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=1, bias=False)  # 1x1 conv, expands channels by a further 2x (4x total)
        self.downsample = downsample  # flag: whether to project the identity path (immediately overwritten below with the projection module itself)
        self.b3 = nn.BatchNorm2d(planes * 4)  # normalize after c3
        self.downsample = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),  # 1x1 conv projecting identity to match c3's expanded channel count
            nn.BatchNorm2d(planes * 4),  # normalize the projected identity
        )  # note: this reassigns self.downsample from a bool to a module, so it is always applied in forward
        self.relu = nn.ReLU(inplace=True)  # shared activation used after each conv and after the residual add

    def forward(self, x):
        identity = x  # keep original input for the residual connection

        out = self.c1(x)  # 1x1 conv
        out = self.b1(out)  # batchnorm
        out = self.relu(out)  # activation

        out = self.c2(out)  # kxk conv, channel expansion x2
        out = self.b2(out)  # batchnorm
        out = self.relu(out)  # activation

        out = self.c3(out)  # 1x1 conv, channel expansion x2 (x4 total vs input)
        out = self.b3(out)  # batchnorm

        if self.downsample:  # self.downsample is the Sequential module set in __init__, always truthy
            identity = self.downsample(x)  # project identity to match out's channel count
        out += identity  # residual addition
        out = self.relu(out)  # final activation

        return out  # (B, planes*4, H, W)


class ResidualBlock1D_(nn.Module):

    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D_, self).__init__()  # standard nn.Module init
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)  # 1x1 conv, channels unchanged
        self.b1 = nn.BatchNorm1d(planes)  # normalize after c1
        self.c2 = nn.Conv1d(planes, planes * 2, kernel_size=11, stride=1,
                            padding=5, bias=False)  # kernel-11 conv, expands channels by 2x, length-preserving (padding=5)
        self.b2 = nn.BatchNorm1d(planes * 2)  # normalize after c2
        self.c3 = nn.Conv1d(planes * 2, planes * 8, kernel_size=1, stride=1, bias=False)  # 1x1 conv, expands channels by 4x further (8x total)
        self.b3 = nn.BatchNorm1d(planes * 8)  # normalize after c3
        self.downsample = nn.Sequential(
            nn.Conv1d(planes, planes * 8, kernel_size=1, stride=1, bias=False),  # 1x1 conv projecting identity to match c3's expanded channel count
            nn.BatchNorm1d(planes * 8),  # normalize the projected identity
        )  # always constructed and always applied below regardless of the `downsample` argument
        self.relu = nn.ReLU(inplace=True)  # shared activation used after each conv and after the residual add

    def forward(self, x):
        identity = x  # keep original input for the residual connection

        out = self.c1(x)  # 1x1 conv
        out = self.b1(out)  # batchnorm
        out = self.relu(out)  # activation

        out = self.c2(out)  # kernel-11 conv, channel expansion x2
        out = self.b2(out)  # batchnorm
        out = self.relu(out)  # activation

        out = self.c3(out)  # 1x1 conv, channel expansion x4 (x8 total vs input)
        out = self.b3(out)  # batchnorm

        if self.downsample:  # self.downsample is the Sequential module set in __init__ (the `downsample` constructor arg is otherwise unused), always truthy
            identity = self.downsample(x)  # project identity to match out's channel count

        out += identity  # residual addition
        out = self.relu(out)  # final activation

        return out  # (B, planes*8, L)


class ResidualBlock1D(nn.Module):
    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D, self).__init__()  # standard nn.Module init
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)  # 1x1 conv, channels unchanged
        self.b1 = nn.BatchNorm1d(planes)  # normalize after c1
        self.c2 = nn.Conv1d(planes, planes, kernel_size=11, stride=1,  # kernel 11
                            padding=5, bias=False)  # kernel-11 conv, channels unchanged, length-preserving (padding=5)
        self.b2 = nn.BatchNorm1d(planes)  # normalize after c2
        self.c3 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)  # 1x1 conv, channels unchanged
        self.b3 = nn.BatchNorm1d(planes)  # normalize after c3
        self.downsample = downsample  # flag controlling whether the (optional) projection shortcut is used
        if downsample:
            self.down_sample = nn.Sequential(
                nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False),  # 1x1 conv identity projection (channels already match here)
                nn.BatchNorm1d(planes),  # normalize the projected identity
            )  # only constructed when downsample=True, unlike the `_` variants above
        self.relu = nn.ReLU(inplace=True)  # shared activation used after each conv and after the residual add

    def forward(self, x):
        identity = x  # keep original input for the residual connection

        out = self.c1(x)  # 1x1 conv
        out = self.b1(out)  # batchnorm
        out = self.relu(out)  # activation

        out = self.c2(out)  # kernel-11 conv
        out = self.b2(out)  # batchnorm
        out = self.relu(out)  # activation

        out = self.c3(out)  # 1x1 conv
        out = self.b3(out)  # batchnorm

        if self.downsample:
            identity = self.down_sample(x)  # project identity through the learned shortcut

        out += identity  # residual addition
        out = self.relu(out)  # final activation

        return out  # (B, planes, L)


class ResidualBlock2D(nn.Module):
    def __init__(self, planes, kernel_size=(11, 5), padding=(5, 2), downsample=True):
        super(ResidualBlock2D, self).__init__()  # standard nn.Module init
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)  # 1x1 conv, channels unchanged
        self.b1 = nn.BatchNorm2d(planes)  # normalize after c1
        self.c2 = nn.Conv2d(planes, planes * 2, kernel_size=kernel_size, stride=1,
                            padding=padding, bias=False)  # kxk conv, expands channels by 2x
        self.b2 = nn.BatchNorm2d(planes * 2)  # normalize after c2
        self.c3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=1, bias=False)  # 1x1 conv, expands channels by a further 2x (4x total)
        self.b3 = nn.BatchNorm2d(planes * 4)  # normalize after c3
        self.downsample = downsample  # flag controlling whether the projection shortcut is used
        self.down_sample = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),  # 1x1 conv projecting identity to match c3's expanded channel count
            nn.BatchNorm2d(planes * 4),  # normalize the projected identity
        )  # always constructed regardless of the downsample flag (only its use in forward is conditional)
        self.relu = nn.ReLU(inplace=True)  # shared activation used after each conv and after the residual add

    def forward(self, x):
        identity = x  # keep original input for the residual connection

        out = self.c1(x)  # 1x1 conv
        out = self.b1(out)  # batchnorm
        out = self.relu(out)  # activation

        out = self.c2(out)  # kxk conv, channel expansion x2
        out = self.b2(out)  # batchnorm
        out = self.relu(out)  # activation

        out = self.c3(out)  # 1x1 conv, channel expansion x2 (x4 total vs input)
        out = self.b3(out)  # batchnorm

        if self.downsample:
            identity = self.down_sample(x)  # project identity to match out's channel count
        out += identity  # residual addition
        out = self.relu(out)  # final activation

        return out  # (B, planes*4, H, W)
