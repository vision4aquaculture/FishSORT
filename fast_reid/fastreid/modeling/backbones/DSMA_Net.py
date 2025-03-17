# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

import numpy as np

from fast_reid.fastreid.layers import (
    IBN,
    SELayer,
    Non_local,
    get_norm,
)
from fast_reid.fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from fast_reid.fastreid.utils import comm

logger = logging.getLogger(__name__)
model_urls = {

}


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class DSM_SpatialGateIROS(nn.Module):
    def __init__(self, channel):
        super(DSM_SpatialGateIROS, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, kernel_size, act=False)
        self.dw1 = nn.Sequential(
            Conv(channel, channel, 5, s=1, d=2, g=channel, act=nn.GELU()),
            Conv(channel, channel, 7, s=1, d=3, g=channel, act=nn.GELU())
        )
        # self.dw2 = Conv(channel, channel, kernel_size, g=channel, act=nn.GELU())      #original
        self.dw2 = Conv(channel, channel, kernel_size, g=channel, act=nn.GELU())

    def forward(self, x, y):
        out = self.compress(y)  # v过门控，得到空间掩码
        out = self.spatial(out)
        # print(f"output shape to DSMAttention: {out.shape}")  # 打印输入类型
        # print(f"y shape to DSMAttention: {y.shape}")  # 打印输入类型
        # print(f"self.dw2(x) shape to DSMAttention: {self.dw2(x).shape}")  # 打印输入类型
        out = y * out
        out = out + self.dw2(x)
        # out = x * out + self.dw2(x)
        return out


class DSM_LocalAttention(nn.Module):
    def __init__(self, channel, p) -> None:
        super().__init__()
        self.channel = channel

        self.num_patch = 2 ** p
        self.sig = nn.Sigmoid()

        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2, 3), keepdim=True)
        # print(f"output shape to DSMAttention: {out.shape}")  # 打印输入类型
        # print(f"self.a shape to DSMAttention: {self.a.shape}")  # 打印输入类型
        # print(f"self.b shape to DSMAttention: {self.b.shape}")  # 打印输入类型
        # print(f"x shape to DSMAttention: {x.shape}")  # 打印输入类型
        a = self.a * out * x
        b = self.b * x
        out = a + b
        return out
        # return self.a * out * x + self.b * x


class DualDomainSelectionMechanism_IROS(nn.Module):  # 双域选择机制，关键，用于qkv点积结果过门控
    # https://openaccess.thecvf.com/content/ICCV2023/papers/Cui_Focal_Network_for_Image_Restoration_ICCV_2023_paper.pdf
    # https://github.com/c-yn/FocalNet
    # Dual-DomainSelectionMechanism
    def __init__(self, channel) -> None:
        super().__init__()
        pyramid = 1
        self.spatial_gate_IROS = DSM_SpatialGateIROS(channel)
        layers = [DSM_LocalAttention(channel, p=i) for i in range(pyramid - 1, -1, -1)]
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel, 1, 1))
        self.b = nn.Parameter(torch.ones(channel, 1, 1))

    def forward(self, x, y):
        out = self.spatial_gate_IROS(x, y)
        out = self.local_attention(out)
        return self.a * out + self.b * x


class ScharrConv(nn.Module):
    def __init__(self, channel):
        super(ScharrConv, self).__init__()

        # 定义Scharr算子的水平和垂直卷积核
        scharr_kernel_x = np.array([[3, 0, -3],
                                    [10, 0, -10],
                                    [3, 0, -3]], dtype=np.float32)

        scharr_kernel_y = np.array([[3, 10, 3],
                                    [0, 0, 0],
                                    [-3, -10, -3]], dtype=np.float32)

        # 将Scharr核转换为PyTorch张量并扩展为通道数
        scharr_kernel_x = torch.tensor(scharr_kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        scharr_kernel_y = torch.tensor(scharr_kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

        # 扩展为多通道
        self.scharr_kernel_x = scharr_kernel_x.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)
        self.scharr_kernel_y = scharr_kernel_y.expand(channel, 1, 3, 3)  # (channel, 1, 3, 3)

        # 定义卷积层，但不学习卷积核，直接使用Scharr核
        self.scharr_kernel_x_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.scharr_kernel_y_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        # 将卷积核的权重设置为Scharr算子的核
        self.scharr_kernel_x_conv.weight.data = self.scharr_kernel_x.clone()
        self.scharr_kernel_y_conv.weight.data = self.scharr_kernel_y.clone()

        # 禁用梯度更新
        self.scharr_kernel_x_conv.requires_grad = False
        self.scharr_kernel_y_conv.requires_grad = False

    def forward(self, x):
        # 对输入的特征图进行Scharr卷积（水平和垂直方向）
        grad_x = self.scharr_kernel_x_conv(x)
        grad_y = self.scharr_kernel_y_conv(x)

        # 计算梯度幅值
        edge_magnitude = grad_x * 0.5 + grad_y * 0.5

        return edge_magnitude


class FreqSpatial(nn.Module):
    def __init__(self, in_channels):
        super(FreqSpatial, self).__init__()

        self.sed = ScharrConv(in_channels)

        # 时域卷积部分
        self.spatial_conv1 = Conv(in_channels, in_channels)
        self.spatial_conv2 = Conv(in_channels, in_channels)

        # 频域卷积部分
        self.fft_conv = Conv(in_channels * 2, in_channels * 2, 3)
        self.fft_conv2 = Conv(in_channels, in_channels, 3)

        self.final_conv = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        batch, c, h, w = x.size()
        # 时域提取
        spatial_feat = self.sed(x)
        spatial_feat = self.spatial_conv1(spatial_feat)
        spatial_feat = self.spatial_conv2(spatial_feat + x)

        # 频域卷积
        # 1. 先转换到频域
        fft_feat = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
        fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w').contiguous()

        # 2. 频域卷积处理
        fft_feat = self.fft_conv(fft_feat)

        # 3. 还原回时域
        fft_feat = rearrange(fft_feat, 'b (c d) h w -> b c h w d', d=2).contiguous()
        fft_feat = torch.view_as_complex(fft_feat)
        fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm='ortho')

        fft_feat = self.fft_conv2(fft_feat)

        # 合并时域和频域特征
        out = spatial_feat + fft_feat
        return self.final_conv(out)


class FreqSpatialFilter(FreqSpatial):  # 频域滤波器,导入频域特征
    def __init__(self, in_channels, cutoff=30, fw=0.5):
        super(FreqSpatialFilter, self).__init__(in_channels)
        self.cutoff = cutoff
        self.filter_weight = fw
        # self.fft_conv = Conv(in_channels*2, in_channels * 2, 3)
        # self.fft_conv2 = Conv(in_channels * 2, in_channels * 2, 3)
        #
        # self.final_conv = Conv(in_channels, in_channels, 1)

    def forward(self, x):
        batch, c, h, w = x.size()
        device = x.device  # 获取输入张量所在的设备
        x = x.float()
        # 频域卷积
        # 1. 先转换到频域
        fft_feat = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
        fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w').contiguous()

        # 分离高频与低频
        lowpass_filter = self.create_lowpass_filter((h, w), self.cutoff)
        lowpass_filter = lowpass_filter.unsqueeze(0).unsqueeze(0)
        highpass_filter = 1 - lowpass_filter

        # 调整滤波器最后一维的尺寸以匹配 rfft2 的输出,并移入device
        lowpass_filter = lowpass_filter[..., :fft_feat.size(-1)]
        highpass_filter = highpass_filter[..., :fft_feat.size(-1)]

        # 将滤波器移动到与输入张量相同的设备上
        lowpass_filter = lowpass_filter.to(device)
        highpass_filter = highpass_filter.to(device)

        # 扩展滤波器维度以匹配 4 维信号
        lowpass_filter = lowpass_filter.repeat(batch, c * 2, 1, 1)
        highpass_filter = highpass_filter.repeat(batch, c * 2, 1, 1)

        filters = [lowpass_filter, highpass_filter]
        freq_feats = []

        # 对高频与低频批量处理
        for filter in filters:
            # 分离高低频信号
            filtered_fft_feat = fft_feat * filter

            # 2. 频域卷积处理
            filtered_fft_feat = self.fft_conv(filtered_fft_feat)

            # 3. 还原回时域
            filtered_fft_feat = filtered_fft_feat.float()
            filtered_fft_feat = rearrange(filtered_fft_feat, 'b (c d) h w -> b c h w d', d=2).contiguous()
            filtered_fft_feat = torch.view_as_complex(filtered_fft_feat)
            filtered_fft_feat = torch.fft.irfft2(filtered_fft_feat, s=(h, w), norm='ortho')

            filtered_fft_feat = self.fft_conv2(filtered_fft_feat)
            freq_feats.append(filtered_fft_feat)

        low_frequency_fft_feat, high_frequency_fft_feat = freq_feats

        # 合并加权后的频域特征
        out = low_frequency_fft_feat * self.filter_weight + high_frequency_fft_feat * (1 - self.filter_weight)
        out = self.final_conv(out)

        out = out.half()
        return out

    def create_lowpass_filter(self, size, cutoff):
        rows, cols = size
        crow, ccol = rows // 2, cols // 2
        y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols))
        dist = torch.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        mask = (dist <= cutoff).float()
        return mask


class DSMAttention(nn.Module):  # self-attention
    """ Factorized attention with convolutional relative position encoding class. """

    def __init__(self, c1, num_heads=8, fc=256, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__()

        self.num_heads = num_heads
        self.c1 = c1
        head_dim = c1 // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(c1, c1 * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj3 = nn.Linear(c1, c1)
        self.proj_drop = nn.Dropout(proj_drop)

        # embedding
        self.proj1 = nn.Conv2d(c1, c1, kernel_size=1)
        self.proj4 = nn.Conv2d(c1//16, c1, kernel_size=1)
        self.norm1 = nn.LayerNorm(c1)

        # pooling positional encoding
        self.proj2 = nn.AdaptiveAvgPool2d((None, None))

        # self.gap = nn.AdaptiveAvgPool2d((1, fc // num_heads))
        # self.gap2 = nn.AdaptiveAvgPool2d((fc // num_heads, fc // num_heads))
        # self.fc = nn.Sequential(
        #     nn.Linear(fc, fc),
        #     nn.BatchNorm1d(fc),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(fc, fc),
        #     nn.Sigmoid()
        # )
        # 新增的模块
        self.FreqSpatialFilter = FreqSpatialFilter(c1)
        self.dsm = DualDomainSelectionMechanism_IROS(8)

    def forward(self, x):
        # 频域变换
        # print(f"Input type to DSMAttention: {x.dtype}")  # 打印输入类型
        # print(f"Input x shape: {x.shape}")  # 添加打印语句
        f = self.FreqSpatialFilter(x)  # [64,32,96,32]

        # embedding
        _, _, H, W = x.shape
        x = self.proj1(x).flatten(2).transpose(1, 2)
        # print(f"Input x shape to DSMAttention: {x.shape}")  # 添加打印语句
        x = self.norm1(x)  # [16, 400, 256]

        f = self.proj1(f).flatten(2).transpose(1, 2)
        f = self.norm1(f)  # [16, 400, 256]

        # Pooling positional encoding.
        B, N, C = x.shape
        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj2(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape
        # print(f"Input f shape: {f.shape}")  # 添加打印语句
        # Generate Q, K, V.# Shape: [3, B, h, N, Ch].
        qkvt = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkvf = self.qkv(f).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qt, kt, vt = qkvt[0], qkvt[1], qkvt[2]  # Shape: [B, h, N, Ch].
        qf, kf, vf = qkvf[0], qkvf[1], qkvf[2]  # Shape: [B, h, N, Ch].
        q, k, v = qt + qf, kt + kf, vt + vf  # Shape: [B, h, N, Ch].    #可加权
        # q, k, v = qt, kt, vt # Shape: [B, h, N, Ch].    #可加权
        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)  # Shape: [B, h, Ch, Ch]. #k，v点积结果
        factor_att = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)  # Shape: [B, h, N, Ch]. #q与kv点积结果
        # factor_att_softmax = factor_att.softmax(dim=2)  # Softmax on dim N.
        # v_softmax = v.softmax(dim=2)  # Softmax on dim N.
        # print(f"factor_att shape to DSMAttention: {factor_att.shape}")  # 打印输入形状
        # print(f"v shape to DSMAttention: {v.shape}")  # 打印输入形状
        # factor_att = factor_att.reshape(B, C, H, W)  # [64, 8, 3072, 8]
        # v = v.reshape(B, C, H, W)  # [64, 8, 3072, 8]
        x = self.dsm(factor_att, v)  # [64, 8, 3072, 8]
        # x = x.transpose(1, 2).view(B, C, H, W)

        # x = k_softmax_T_dot_v
        # x = torch.abs(x)
        # x_abs = x
        # x = self.gap(x)
        # x = torch.flatten(x, 1)
        # average = x
        #
        # x = self.fc(x)
        # x = torch.mul(average, x)
        # x = x.view(B, self.num_heads, C // self.num_heads)
        # x = x.unsqueeze(2)
        # # soft thresholding
        # sub = x_abs - x
        # zeros = sub - sub
        # n_sub = torch.max(sub, zeros)
        # x = (torch.sign(factor_att) @ n_sub) + v

        # Output projection.
        x = self.scale * x
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection.
        x = self.proj3(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        return x


######################################## DSMAttention end ########################################

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class MutilScaleEdgeInformationSelect(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()

        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.dsm = DualDomainSelectionMechanism_IROS(inc * 2)
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):
        print("in_channels:", self.c)
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(self.dsm(torch.cat(out, 1)))


class CSP_MutilScaleEdgeInformationSelect(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        # self.m = nn.ModuleList(MutilScaleEdgeInformationSelect(self.c, [3, 6, 9, 12]) for _ in range(n))
        self.m = nn.ModuleList(DSMAttention(self.c) for _ in range(n))


class DSMA_Net(nn.Module):  # backbone
    def __init__(self, c1=3, c2=512, k=3, s=2):
        super().__init__()
        # self.c1=c1
        # self.c2=c2
        # self.k=k
        # self.s=s
        self.conv1 = Conv(3, 64, 3, 2)
        self.conv2 = Conv(64, 128, 3, 2)
        self.DSMAttention1 = CSP_MutilScaleEdgeInformationSelect(128, 128, 1)
        self.conv3 = Conv(128, 256, 3, 2)
        self.DSMAttention2 = CSP_MutilScaleEdgeInformationSelect(256, 256, 1)
        self.conv4 = Conv(256, 384, 3, 2)
        self.DSMAttention3 = CSP_MutilScaleEdgeInformationSelect(384, 384, 1)
        self.conv5 = Conv(384, 384, 3, 2)
        self.DSMAttention4 = CSP_MutilScaleEdgeInformationSelect(384, 384, 1)
        self.DSMAttention5 = CSP_MutilScaleEdgeInformationSelect(384, 384, 1)
        self.DSMAttention6 = CSP_MutilScaleEdgeInformationSelect(384, 384, 1)
        self.conv6 = Conv(384, 2048, 3, 2)

    def forward(self, x):
        # if x.dtype != torch.float32:
        #     x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.DSMAttention1(x)
        x = self.conv3(x)
        x = self.DSMAttention2(x)
        x = self.conv4(x)
        x = self.DSMAttention3(x)
        x = self.conv5(x)
        x = self.DSMAttention4(x)
        x = self.DSMAttention5(x)
        x = self.DSMAttention6(x)
        x = self.conv6(x)
        return x


def init_pretrained_weights(key):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = model_urls[key].split('/')[-1]

    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        logger.info(f"Pretrain model don't exist, downloading from {model_urls[key]}")
        if comm.is_main_process():
            gdown.download(model_urls[key], cached_file, quiet=False)

    comm.synchronize()

    logger.info(f"Loading pretrained model from {cached_file}")
    state_dict = torch.load(cached_file, map_location=torch.device('cpu'))

    return state_dict


@BACKBONE_REGISTRY.register()
def build_DSMANet_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    model = DSMA_Net()
    # model = model.half()
    # model = DSMA_Net(last_stride, bn_norm, with_ibn, with_se, with_nl, block,
    #                num_blocks_per_stage, nl_layers_per_stage, for_rtdetr)

    return model


class ResNet_rtdetr(nn.Module):
    '''
    新建一个类，用于创建适配rtdetrEncoderde的resnet类
    '''

    def __init__(self):
        super().__init__()
        from ultralytics.nn.modules.block import ConvNormLayer, Blocks, BasicBlock
        from torch.nn.modules.pooling import MaxPool2d
        from ultralytics.nn.modules.conv import Conv
        self.CN1 = ConvNormLayer(3, 32, 3, 2, None, False, 'relu')
        self.CN2 = ConvNormLayer(32, 32, 3, 1, None, False, 'relu')
        self.CN3 = ConvNormLayer(32, 64, 3, 1, None, False, 'relu')
        self.pool = MaxPool2d(3, 2, 1)

        self.b1 = Blocks(64, 64, BasicBlock, 2, 2, 'relu')
        self.b2 = Blocks(64, 128, BasicBlock, 2, 3, 'relu')
        self.b3 = Blocks(128, 256, BasicBlock, 2, 4, 'relu')
        self.b4 = Blocks(256, 512, BasicBlock, 2, 5, 'relu')

    def forward(self, x):
        x = self.CN1(x)
        x = self.CN2(x)
        x = self.CN3(x)
        x = self.pool(x)

        x = self.b1(x)
        x = self.b2(x)
        y = self.b3(x)
        z = self.b4(y)

        x = [x, y, z]
        return x
