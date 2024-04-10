"""MCJA/models/modules/mda.py
   It defines the Modality Distribution Adapter (MDA) class, a module designed to enhance cross-modal feature learning.
"""

import torch
import torch.nn as nn


class MDA(nn.Module):
    """
    A module implementing the Modality Distribution Adapter (MDA), a mechanism designed to enhance cross-modal learning
    by adaptively re-weighting feature channels based on their relevance to different modalities. The MDA module
    dynamically adjusts the contribution of different feature channels to the task at hand, depending on the context
    provided by different modalities, thereby facilitating more effective integration of multimodal information.

    Args:
    - in_channels (int): Number of channels in the input feature map.
    - inter_ratio (int): Reduction ratio for intermediate channel dimensions, controlling the compactness of the module.
    - m_num (int): Number of distinct modalities that the model needs to adapt to.

    Methods:
    - forward(x): Processes the input tensor through the MDA to produce an output with adapted feature distributions.
    """

    def __init__(self, in_channels, inter_ratio=2, m_num=2):
        super(MDA, self).__init__()
        self.in_channels = in_channels
        self.inter_ratio = inter_ratio
        self.planes = in_channels // inter_ratio
        self.m_num = m_num

        self.sc_conv = nn.Conv2d(in_channels, m_num, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        self.ca_conv = nn.Conv1d(self.in_channels, self.in_channels,
                                 kernel_size=m_num, groups=self.in_channels, bias=False)
        self.ca_bn = nn.BatchNorm1d(self.in_channels)

        self.norm_bn = nn.BatchNorm2d(self.in_channels)
        nn.init.constant_(self.norm_bn.weight, 0.0)
        nn.init.constant_(self.norm_bn.bias, 0.0)

    def forward(self, x):
        input_x = x
        batch, channel, height, width = x.size()

        # Spatial Characteristics Learning -----------------------------------------------------------------------------
        # [B, C, H, W] -> [B, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [B, C, H * W] -> [B, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [B, C, H, W] -> [B, M, H, W]
        context_mask = self.sc_conv(x)
        # [B, M, H, W] -> [B, M, H * W]
        context_mask = context_mask.view(batch, self.m_num, height * width)
        # [B, M, H * W] -> [B, M, H * W]
        context_mask = self.softmax(context_mask)
        # [B, M, H * W] -> [B, M, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [B, 1, C, H * W] [B, M, H * W, 1] -> [B, M, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [B, M, C, 1] -> [B, C, M]
        context = context.squeeze(-1).permute(0, 2, 1)

        # Characteristics Aggregation ----------------------------------------------------------------------------------
        # [B, C, M] -> [B, C, 1]
        z = self.ca_conv(context)
        # [B, C, 1] -> [B, C, 1]
        z = self.ca_bn(z)
        # [B, C, 1] -> [B, C, 1]
        g = torch.sigmoid(z)

        # Feature Distribution Adaption --------------------------------------------------------------------------------
        # [B, C, 1] -> [B, C, 1, 1]
        g = g.view(batch, channel, 1, 1)
        # [B, C, H, W] [B, C, 1, 1] -> [B, C, H, W]
        out = self.norm_bn(x * g.expand_as(x)) + x

        return out
