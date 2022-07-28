import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional
# from .D3D.modules.deform_conv import *

class RecurrentDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool = AvgPool()

        self.decode4 = BottleneckBlock(in_channels[3])
        self.decode3 = UpsamplingBlock(out_channels[0], in_channels[2], 4, out_channels[1])
        self.decode2 = UpsamplingBlock(out_channels[1], in_channels[1], 4, out_channels[2])
        self.decode1 = UpsamplingBlock(out_channels[2], in_channels[0], 4, out_channels[3])
        self.decode0 = OutputBlock(out_channels[3], 3, out_channels[4])

        self.project_OS1 = nn.Sequential(
            nn.Conv2d(out_channels[4], 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
        )
        self.project_OS4 = nn.Sequential(
            nn.Conv2d(out_channels[2], 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        self.project_OS8 = nn.Sequential(
            nn.Conv2d(out_channels[1], 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )
        self.project_seg = nn.Sequential(
            nn.Conv2d(out_channels[4], 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor,
                r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor],
                segmentation_pass: bool = False):
        # x: output from the previous stage
        # f: output from the encoder block
        # s: downsampled source image after average pooling
        # r: output from the previous stage (hidden state)
        if not segmentation_pass:
            time_series = True if s0.ndim == 5 else False

            s1, s2, s3 = self.avgpool(s0)
            x4, r4 = self.decode4(f4, r4)
            x3, r3 = self.decode3(x4, f3, s3, r3)

            if time_series:
                B, T = x3.shape[:2]
                os8 = self.project_OS8(x3.flatten(0, 1))
                os8 = F.interpolate(os8, scale_factor=8.0, mode='bilinear', align_corners=False)
                os8 = os8.unflatten(0, (B, T))

                x2, r2 = self.decode2(x3, f2, s2, r2)

                B, T = x2.shape[:2]
                os4 = self.project_OS4(x2.flatten(0, 1))
                os4 = F.interpolate(os4, scale_factor=4.0, mode='bilinear', align_corners=False)
                os4 = os4.unflatten(0, (B, T))

                x1, r1 = self.decode1(x2, f1, s1, r1)
                x0 = self.decode0(x1, s0)

                B, T = x0.shape[:2]
                os1 = self.project_OS1(x0.flatten(0, 1))
                os1 = os1.unflatten(0, (B, T))
                
                os8 = os8[:, :, :, :os1.size(3), :os1.size(4)]
                os4 = os4[:, :, :, :os1.size(3), :os1.size(4)]
            else:
                B, T = x3.shape[0], 1
                os8 = self.project_OS8(x3)
                os8 = F.interpolate(os8, scale_factor=8.0, mode='bilinear', align_corners=False)

                x2, r2 = self.decode2(x3, f2, s2, r2)

                B, T = x2.shape[0], 1
                os4 = self.project_OS4(x2)
                os4 = F.interpolate(os4, scale_factor=4.0, mode='bilinear', align_corners=False)

                x1, r1 = self.decode1(x2, f1, s1, r1)
                x0 = self.decode0(x1, s0)

                B, T = x0.shape[0], 1
                os1 = self.project_OS1(x0)

                os8 = os8[:, :, :os1.size(2), :os1.size(3)]
                os4 = os4[:, :, :os1.size(2), :os1.size(3)]                

            output = {
                'hid':   x0,
                'os1':  os1,
                'os4':  os4,
                'os8':  os8
            }

            return output, r1, r2, r3, r4
        else:
            s1, s2, s3 = self.avgpool(s0)
            x4, r4 = self.decode4(f4, r4)
            x3, r3 = self.decode3(x4, f3, s3, r3)
            x2, r2 = self.decode2(x3, f2, s2, r2)
            x1, r1 = self.decode1(x2, f1, s1, r1)
            x0 = self.decode0(x1, s0)

            B, T = x0.shape[:2]
            seg = self.project_seg(x0.flatten(0, 1))
            seg = seg.unflatten(0, (B, T))

            output = {
                'seg':  seg
            }

            return output, r1, r2, r3, r4
    

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
        # self.gru = ConvGRU(channels // 2)
        self.gru = ConvGRU(channels)
        # self.attention = SelfAttention(channels, True)
        # self.deform = DeformableConvolution(channels)

    def forward(self, x, r: Optional[Tensor]):
        # x = self.attention(x)
        # with torch.cuda.amp.autocast(enabled=False):
        #     x = x.float()
        #     x = self.deform(x)
        # a, b = x.split(self.channels // 2, dim=-3)
        # b, r = self.gru(b, r)
        # x = torch.cat([a, b], dim=-3)
        x, r = self.gru(x, r)
        return x, r

    
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.attent = CoordAtt(out_channels, out_channels)
        # self.gru = ConvGRU(out_channels // 2)
        self.gru = ConvGRU(out_channels)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        # x = torch.cat([x, f, s[:, :3, :, :]], dim=1)
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        # x = self.attent(x)
        # a, b = x.split(self.out_channels // 2, dim=1)
        # b, r = self.gru(b, r)
        # x = torch.cat([a, b], dim=1)
        x, r = self.gru(x, r)
        return x, r

    def forward_time_series(self, x, f, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        # x = torch.cat([x, f, s[:, :3, :, :]], dim=1)
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        # x = self.attent(x)
        x = x.unflatten(0, (B, T))
        # a, b = x.split(self.out_channels // 2, dim=2)
        # b, r = self.gru(b, r)
        # x = torch.cat([a, b], dim=2)
        x, r = self.gru(x, r)
        # x = self.attention(x)
        # with torch.cuda.amp.autocast(enabled=False):
        #     x = x.float()
        #     x = self.deform(x)
        return x, r


    def forward(self, x, f, s, r: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)


class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.attent = CoordAtt(out_channels, out_channels)
        
    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s[:, :3, :, :]], dim=1)
        # x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x
    
    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s[:, :3, :, :]], dim=1)
        # x = torch.cat([x, s], dim=1)
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
    