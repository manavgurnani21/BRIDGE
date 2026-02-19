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

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel * reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return y


class SEBlock_(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SEBlock_, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class ResidualBlock2D_(nn.Module):

    def __init__(self, planes, kernel_size=(11, 5), padding=(5, 2), downsample=True):
        super(ResidualBlock2D_, self).__init__()
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes * 2, kernel_size=kernel_size, stride=1,
                            padding=padding, bias=False)
        self.b2 = nn.BatchNorm2d(planes * 2)
        self.c3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=1, bias=False)
        self.downsample = downsample
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResidualBlock1D_(nn.Module):

    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D_, self).__init__()
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes, planes * 2, kernel_size=11, stride=1,
                            padding=5, bias=False)
        self.b2 = nn.BatchNorm1d(planes * 2)
        self.c3 = nn.Conv1d(planes * 2, planes * 8, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm1d(planes * 8)
        self.downsample = nn.Sequential(
            nn.Conv1d(planes, planes * 8, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(planes * 8),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualBlock1D(nn.Module):
    def __init__(self, planes, downsample=True):
        super(ResidualBlock1D, self).__init__()
        self.c1 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm1d(planes)
        self.c2 = nn.Conv1d(planes, planes, kernel_size=11, stride=1,  # kernel 11
                            padding=5, bias=False)
        self.b2 = nn.BatchNorm1d(planes)
        self.c3 = nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        if downsample:
            self.down_sample = nn.Sequential(
                nn.Conv1d(planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(planes),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualBlock2D(nn.Module):
    def __init__(self, planes, kernel_size=(11, 5), padding=(5, 2), downsample=True):
        super(ResidualBlock2D, self).__init__()
        self.c1 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False)
        self.b1 = nn.BatchNorm2d(planes)
        self.c2 = nn.Conv2d(planes, planes * 2, kernel_size=kernel_size, stride=1,
                            padding=padding, bias=False)
        self.b2 = nn.BatchNorm2d(planes * 2)
        self.c3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, stride=1, bias=False)
        self.b3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.down_sample = nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.c1(x)
        out = self.b1(out)
        out = self.relu(out)

        out = self.c2(out)
        out = self.b2(out)
        out = self.relu(out)

        out = self.c3(out)
        out = self.b3(out)

        if self.downsample:
            identity = self.down_sample(x)
        out += identity
        out = self.relu(out)

        return out
