import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional
from .D3D.modules.deform_conv import *

class RecurrentDecoder(nn.Module):
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor):
        # x: output from the previous stage
        # f: output from the encoder block
        # s: downsampled feature map after average pooling
        # r: output from the previous stage (hidden state)
        s1, s2, s3 = self.avgpool(s0)
        x4 = self.decode4(f4)
        x3 = self.decode3(x4, f3, s3)
        x2 = self.decode2(x3, f2, s2)
        x1 = self.decode1(x2, f1, s1)
        x0 = self.decode0(x1, s0)
        return x0


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_q = nn.Conv3d(channels, channels // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_k = nn.Conv3d(channels, channels // 8, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv_v = nn.Conv3d(channels, channels, kernel_size=(1, 1, 1))
        self.softmax = nn.Softmax(2)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        b, t, c, h, w = x.size()
        x = torch.permute(x, (0, 2, 1, 3, 4))
        
        x_q = self.conv_q(x).permute(0, 2, 1, 3, 4).flatten(2, 4)
        x_k = self.conv_k(x).permute(0, 2, 1, 3, 4).flatten(2, 4)
        x_v = self.conv_v(x).permute(0, 2, 1, 3, 4).flatten(2, 4)

        x_k = torch.transpose(x_k, dim0=1, dim1=2)
        energy = torch.bmm(x_q, x_k)
        heat = self.softmax(energy)

        out = torch.bmm(heat, x_v)
        out = out.view(b, t, c, h, w)
        out = self.gamma * out + x.permute(0, 2, 1, 3, 4)

        return out


class DeformableConvolution(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = DeformConvPack(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1, 3, 4))
        x = x.contiguous()
        x_conv = self.conv(x)
        out = self.gamma * x_conv.permute(0, 2, 1, 3, 4) + x.permute(0, 2, 1, 3, 4)
        return out

    

class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)
        
    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3
    
    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        s3 = s3.unflatten(0, (B, T))
        return s1, s2, s3
    
    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # self.attention = SelfAttention(channels)
        self.deform = DeformableConvolution(channels)

    def forward(self, x):
        # x = self.attention(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            x = self.deform(x)
        return x

    
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.attention = SelfAttention(out_channels)
        # self.deform = DeformableConvolution(out_channels)

    def forward_single_frame(self, x, f, s):
        size = list(x.size())
        size[2:4] = [value*2 for value in size[2:4]]
        x = self.upsample(x, output_size = size)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        return x

    def forward_time_series(self, x, f, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        size = list(x.size())
        size[2:4] = [value*2 for value in size[2:4]]
        x = self.upsample(x, output_size = size)
        x = x[:, :, :H, :W]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        x = self.attention(x)
        # with torch.cuda.amp.autocast(enabled=False):
        #     x = x.float()
        #     x = self.deform(x)
        return x


    def forward(self, x, f, s):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s)
        else:
            return self.forward_single_frame(x, f, s)


class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
        
    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h
    
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h
        
    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward_single_frame(self, x):
        return self.conv(x)
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))
        
    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
    