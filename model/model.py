from xml.dom.minidom import Element
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List
import numpy as np

from .shufflenetv2 import ShuffleNetV2Encoder
from .mobilenetv3 import MobileNetV3LargeEncoder
from .micronet import MicroNetEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, SEBlock, Projection
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner
from .swin_transformer import SwinTransformerEncoder

class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'shufflenetv2', 'micronet', 'resnet50', 'swin_transformer']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.se = nn.ModuleList([SEBlock(16), SEBlock(24), SEBlock(40), SEBlock(960)]) 
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        elif variant == 'shufflenetv2':
            self.backbone = ShuffleNetV2Encoder(pretrained_backbone)
            self.se = nn.ModuleList([SEBlock(24), SEBlock(116), SEBlock(232), SEBlock(1024)]) 
            self.aspp = LRASPP(1024, 128)
            self.decoder = RecurrentDecoder([24, 116, 232, 128], [80, 40, 32, 16])
        elif variant == 'micronet':
            self.backbone = MicroNetEncoder(pretrained_backbone)
            self.se = nn.ModuleList([SEBlock(16), SEBlock(24), SEBlock(80), SEBlock(864)]) 
            self.aspp = LRASPP(864, 128)
            self.decoder = RecurrentDecoder([16, 24, 80, 128], [80, 40, 32, 16])
        elif variant == 'resnet50':
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.se = nn.ModuleList([SEBlock(64), SEBlock(256), SEBlock(512), SEBlock(2048)]) 
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])
        else:
            self.backbone = SwinTransformerEncoder()
            self.se = nn.ModuleList([SEBlock(96), SEBlock(192), SEBlock(384), SEBlock(768)]) 
            self.aspp = LRASPP(768, 128)
            self.decoder = RecurrentDecoder([96, 192, 384, 128], [80, 40, 32, 16])

        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()

    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        
        f1, f2, f3, f4 = self.backbone(src_sm)
        f1, f2, f3, f4 = self.se[0](f1), self.se[1](f2), self.se[2](f3), self.se[3](f4)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        
        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]


    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x


class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
		super(NLayerDiscriminator, self).__init__()
		self.getIntermFeat = getIntermFeat
		self.n_layers = n_layers

		kw = 4
		padw = int(np.ceil((kw-1.0)/2))
		sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

		nf = ndf
		for n in range(1, n_layers):
			nf_prev = nf
			nf = min(nf * 2, 512)
			sequence += [[
				nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
				norm_layer(nf), nn.LeakyReLU(0.2, True)
			]]

		nf_prev = nf
		nf = min(nf * 2, 512)
		sequence += [[
			nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
			norm_layer(nf),
			nn.LeakyReLU(0.2, True)
		]]

		sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

		if use_sigmoid:
			sequence += [[nn.Sigmoid()]]

		# if getIntermFeat:
		# 	for n in range(len(sequence)):
		# 		setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
		# else:
		sequence_stream = []
		for n in range(len(sequence)):
			sequence_stream += sequence[n]
		self.model = nn.Sequential(*sequence_stream)

	def forward(self, input):
		# if self.getIntermFeat:
		# 	res = [input]
		# 	for n in range(self.n_layers+2):
		# 		model = getattr(self, 'model'+str(n))
		# 		res.append(model(res[-1]))
		# 	return res[1:]
		# else:
		if input.ndim == 5:
			output = []
			for i in range(input.shape[1]):
				output.append(self.model(input[:, i, :, :, :]))
			return torch.cat(output, 1)
		else:
			return self.model(input)