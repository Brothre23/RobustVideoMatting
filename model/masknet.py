from dis import dis
from xml.dom.minidom import Element
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List
import numpy as np

# from .shufflenetv2 import ShuffleNetV2Encoder
# from .mobilenetv3 import MobileNetV3LargeEncoder
# from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import Projection

from torch import distributed as dist
import segmentation_models_pytorch as smp

from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidualConfig
from torchvision.transforms.functional import normalize


class MaskEncoder(MobileNetV3):
    def __init__(self, pretrained: bool = False):
        super().__init__(
            inverted_residual_setting=[
                InvertedResidualConfig( 16, 3,  16,  16, False, "RE", 1, 1, 1),
                InvertedResidualConfig( 16, 3,  64,  24, False, "RE", 2, 1, 1),  # C1
                InvertedResidualConfig( 24, 3,  72,  24, False, "RE", 1, 1, 1),
                InvertedResidualConfig( 24, 5,  72,  40,  True, "RE", 2, 1, 1),  # C2
                InvertedResidualConfig( 40, 5, 120,  40,  True, "RE", 1, 1, 1),
                InvertedResidualConfig( 40, 5, 120,  40,  True, "RE", 1, 1, 1),
                InvertedResidualConfig( 40, 3, 240,  80, False, "HS", 2, 1, 1),  # C3
                InvertedResidualConfig( 80, 3, 200,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 184,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 184,  80, False, "HS", 1, 1, 1),
                InvertedResidualConfig( 80, 3, 480, 112,  True, "HS", 1, 1, 1),
                InvertedResidualConfig(112, 3, 672, 112,  True, "HS", 1, 1, 1),
                InvertedResidualConfig(112, 5, 672, 160,  True, "HS", 2, 2, 1),  # C4
                InvertedResidualConfig(160, 5, 960, 160,  True, "HS", 1, 2, 1),
                InvertedResidualConfig(160, 5, 960, 160,  True, "HS", 1, 2, 1),
            ],
            last_channel=1280
        )
        
        if pretrained:
            self.load_state_dict(torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'))

        del self.avgpool
        del self.classifier
        
    def forward_single_frame(self, x):
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        x = self.features[0](x)
        x = self.features[1](x)
        # 1/2
        f1 = x
        x = self.features[2](x)
        x = self.features[3](x)
        # 1/4
        f2 = x
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        # 1/8
        f3 = x
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        # 1/16
        f4 = x
        return [f1, f2, f3, f4]
    
    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]
        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


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

    def forward(self, x):
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward_single_frame(self, x, f, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        return x

    def forward_time_series(self, x, f, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, f, s[:, :3, :, :]], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
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
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
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
        x = torch.cat([x, s[:, :3, :, :]], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        return x
    
    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class MaskDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool = AvgPool()

        self.decode4 = BottleneckBlock(in_channels[3])
        self.decode3 = UpsamplingBlock(out_channels[0], in_channels[2], 3, out_channels[1])
        self.decode2 = UpsamplingBlock(out_channels[1], in_channels[1], 3, out_channels[2])
        self.decode1 = UpsamplingBlock(out_channels[2], in_channels[0], 3, out_channels[3])
        self.decode0 = OutputBlock(out_channels[3], 3, out_channels[4])

    def forward(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor):
        # x: output from the previous stage
        # f: output from the encoder block
        # s: downsampled source image after average pooling
        # r: output from the previous stage (hidden state)
        s1, s2, s3 = self.avgpool(s0)

        x4 = self.decode4(f4)
        x3 = self.decode3(x4, f3, s3)
        x2 = self.decode2(x3, f2, s2)
        x1 = self.decode1(x2, f1, s1)
        x0 = self.decode0(x1, s0)

        return x0


class MaskNet(nn.Module):
    def __init__(self,
                 pretrained_backbone: bool = False):
        super().__init__()
        
        self.backbone = MaskEncoder(pretrained_backbone)
        self.aspp = LRASPP(960, 128)
        self.decoder = MaskDecoder([16, 24, 40, 128], [128, 80, 40, 32, 16])
        self.project = Projection(16, 1)

    def forward(self, src: Tensor):

        f1, f2, f3, f4 = self.backbone(src)
        f4 = self.aspp(f4)
        hid = self.decoder(src, f1, f2, f3, f4)

        msk = self.project(hid)

        return msk