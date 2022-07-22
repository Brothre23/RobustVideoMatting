from dis import dis
from xml.dom.minidom import Element
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List
import numpy as np

from .shufflenetv2 import ShuffleNetV2Encoder
from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner
from .PointRend.pointrend import PointRendRefiner

import cv2
from torch import distributed as dist
import segmentation_models_pytorch as smp
from .masknet import MaskNet

class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'point_rend',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'shufflenetv2', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter', 'point_rend']
        
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [128, 80, 40, 32, 16])
        elif variant == 'shufflenetv2':
            self.backbone = ShuffleNetV2Encoder(pretrained_backbone)
            self.aspp = LRASPP(1024, 128)
            self.decoder = RecurrentDecoder([24, 116, 232, 128], [128, 80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [256, 128, 64, 32, 16])

        # self.project_mat = Projection(16, 4)
        # self.project_seg = Projection(16, 1)
        self.kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
        # self.mask_net = smp.UnetPlusPlus(
        #                     encoder_name='timm-mobilenetv3_large_100',
        #                     encoder_weights='imagenet',
        #                     in_channels=3,
        #                     classes=1
        #                 )
        self.mask_net = MaskNet(pretrained_backbone=True)

        if refiner == 'point_rend':
            self.refiner = PointRendRefiner()
        elif refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()

    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                last: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src

        src_sm, msk = self.get_seg_mask(src_sm)
        
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)

        if not segmentation_pass:
            output, r1, r2, r3, r4 = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4, False)

            hid = output['hid']
            os1 = output['os1']
            os4 = output['os4']
            os8 = output['os8']

            if src_sm.ndim == 5:
                fgr_residual, pha_os1 = os1.split([3, 1], dim=2)
                fgr = fgr_residual + src_sm[:, :, :3, :, :]
            else:
                fgr_residual, pha_os1 = os1.split([3, 1], dim=1)
                fgr = fgr_residual + src_sm[:, :3, :, :]
            fgr = fgr.clamp(0., 1.)

            pha_os8 = os8.clamp(0., 1.)
            pha_os4 = os4.clamp(0., 1.)
            pha_os1 = pha_os1.clamp(0., 1.)

            weight_os4 = self.get_unknown_mask(pha_os8, 30)
            pha_os4[weight_os4==0] = pha_os8[weight_os4==0]
            weight_os1 = self.get_unknown_mask(pha_os4, 15)
            pha_os1[weight_os1==0] = pha_os4[weight_os1==0]

            if downsample_ratio != 1:
                # fgr_residual, pha = self.refiner(src[:, :, :3, :, :], src_sm[:, :, :3, :, :], fgr_residual, pha_os1, hid)
                fgr_residual, pha, last = self.refiner(src, hid, fgr_residual, pha_os1, last)

                fgr = fgr_residual + src
                fgr = fgr.clamp(0., 1.)
                pha = pha.clamp(0., 1.)
                last = last.clamp(0., 1.)

                return {
                    'msk':          msk,
                    'seg':          msk,    # none
                    'pha_os1':      msk,    # none
                    'pha_os4':      msk,    # none
                    'pha_os8':      msk,    # none
                    'weight_os1':   msk,    # none
                    'weight_os4':   msk,    # none
                    'fgr':          msk,    # none
                    'pha_lg':       pha,
                    'fgr_lg':       fgr,
                    'r1':           r1,
                    'r2':           r2,
                    'r3':           r3,
                    'r4':           r4,
                    'last':         last,
                }
            else:
                # return [msk, pha_os4, pha_os8, weight_os1, weight_os4, fgr, pha_os1, *rec]
                return {
                    'msk':          msk,
                    'seg':          msk,    # none
                    'pha_os1':      pha_os1,
                    'pha_os4':      pha_os4,
                    'pha_os8':      pha_os8,
                    'weight_os1':   weight_os1,
                    'weight_os4':   weight_os4,
                    'fgr':          fgr,
                    'pha_lg':       msk,    # none
                    'fgr_lg':       msk,    # none
                    'r1':           r1,
                    'r2':           r2,
                    'r3':           r3,
                    'r4':           r4,
                    'last':         msk,    # none
                }
        else:
            output, r1, r2, r3, r4 = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4, True)
            seg = output['seg']
            # return [msk, seg, *rec]
            return {
                'msk':              msk,
                'seg':              seg,
                'pha_os1':          msk,    # none
                'pha_os4':          msk,    # none
                'pha_os8':          msk,    # none
                'weight_os1':       msk,    # none
                'weight_os4':       msk,    # none
                'fgr':              msk,    # none
                'pha_lg':           msk,    # none
                'fgr_lg':           msk,    # none
                'r1':               r1,
                'r2':               r2,
                'r3':               r3,
                'r4':               r4,
                'last':             msk     # none
            }

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

    @torch.jit.ignore
    def get_unknown_mask(self, pred: Tensor, rand_width: int) -> Tensor:
        time_series = True if pred.ndim == 5 else False

        if time_series:
            B, T = pred.shape[:2]
            pred = pred.flatten(0, 1)
        else:
            B, T = pred.shape[0], 1

        pred_np = pred.data.cpu().numpy()
        uncertain_area = np.ones_like(pred_np, dtype=np.uint8)
        uncertain_area[pred_np == 1.0] = 0
        uncertain_area[pred_np == 0.0] = 0

        if self.training:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        # width = rand_width // 2
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))

        for i in range(B * T):
            image = uncertain_area[i, 0, :, :]
            image = cv2.dilate(image, self.kernels[width])
            # image = cv2.dilate(image, kernel)
            uncertain_area[i, 0, :, :] = image

        weight = torch.from_numpy(uncertain_area).to(pred.device)
        if time_series:
            weight = weight.unflatten(0, (B, T))

        return weight

    def get_seg_mask(self, src):
        time_series = True if src.ndim == 5 else False

        if time_series:
            B, T = src.shape[:2]
            src = src.flatten(0, 1)
        else:
            B, T = src.shape[0], 1
        
        msk = self.mask_net(F.interpolate(src, scale_factor=0.5,
                                          mode='bilinear', align_corners=False, recompute_scale_factor=False))
        msk = msk.clamp(0., 1.)
        msk = F.interpolate(msk, size=(src.size(2), src.size(3)),
                            mode='bilinear', align_corners=False, recompute_scale_factor=False)

        if time_series:
            msk = msk.unflatten(0, (B, T))
            src = src.unflatten(0, (B, T))
            src = torch.cat((src, msk), dim=2)
        else:
            src = torch.cat((src, msk), dim=1)  

        return src, msk
