from matplotlib import image
import torch
from torch.nn import functional as F
import numpy as np
import cv2
import kornia

# --------------------------------------------------------------------------------- Train Loss


def lr_matting_loss(pred_msk, pred_fgr, pred_pha_os1, pred_pha_os4, pred_pha_os8, weight_os1, weight_os4, true_fgr, true_pha):
    loss = {}
    loss['total'] = 0.0

    loss['pha_l1'] = (F.l1_loss(pred_pha_os1 * weight_os1, true_pha * weight_os1) * 3 + \
                      F.l1_loss(pred_pha_os4 * weight_os4, true_pha * weight_os4) * 2 + \
                      F.l1_loss(pred_pha_os8, true_pha) * 1) / 6
    loss['pha_coherence'] = (coherence_loss(pred_pha_os1, true_pha) * 3 + \
                             coherence_loss(pred_pha_os4, true_pha) * 2 + \
                             coherence_loss(pred_pha_os8, true_pha) * 1) / 6
    loss['pha_laplacian'] = (laplacian_loss(pred_pha_os1.flatten(0, 1), true_pha.flatten(0, 1), weight_os1.flatten(0, 1)) * 3 + \
                             laplacian_loss(pred_pha_os4.flatten(0, 1), true_pha.flatten(0, 1), weight_os4.flatten(0, 1)) * 2 + \
                             laplacian_loss(pred_pha_os8.flatten(0, 1), true_pha.flatten(0, 1)) * 1) / 6
    loss['msk'] = (F.l1_loss(pred_msk.flatten(0, 1), true_pha.flatten(0, 1)) + laplacian_loss(pred_msk.flatten(0, 1), true_pha.flatten(0, 1))) * 0.25

    true_fg_msk = true_pha.gt(0)
    pred_fgr = pred_fgr * true_fg_msk
    true_fgr = true_fgr * true_fg_msk
    loss['fgr_l1'] = F.l1_loss(true_fgr, pred_fgr) * 2
    loss['fgr_coherence'] = coherence_loss(true_fgr, pred_fgr) * 2

    for key in loss.keys():
        loss['total'] += loss[key]

    return loss    


def hr_matting_loss(pred_fgr, pred_pha, true_fgr, true_pha):
    loss = {}
    loss['total'] = 0.0

    loss['pha_l1'] = F.l1_loss(pred_pha, true_pha)
    loss['pha_coherence'] = coherence_loss(pred_pha, true_pha)
    loss['pha_laplacian'] = laplacian_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1))

    # true_pha_msk = true_pha.flatten(0, 1)
    # true_pha_msk = F.interpolate(true_pha_msk, scale_factor=downsample_ratio,
    #                              mode='bilinear', align_corners=False, recompute_scale_factor=False)
    # loss['msk'] = (F.l1_loss(pred_msk.flatten(0, 1), true_pha_msk) + laplacian_loss(pred_msk.flatten(0, 1), true_pha_msk)) * 0.25

    true_fg_msk = true_pha.gt(0)
    pred_fgr = pred_fgr * true_fg_msk
    true_fgr = true_fgr * true_fg_msk
    loss['fgr_l1'] = F.l1_loss(true_fgr, pred_fgr) * 2
    loss['fgr_coherence'] = coherence_loss(true_fgr, pred_fgr) * 2

    for key in loss.keys():
        loss['total'] += loss[key]

    return loss  


def coherence_loss(pred, true):
    return F.mse_loss(pred[:, 1:] - pred[:, :-1],
                      true[:, 1:] - true[:, :-1]) * 5


def segmentation_loss(pred_msk, pred_seg, true_seg):
    """
    Args:
        pred_seg: Shape(B, T, 1, H, W)
        true_seg: Shape(B, T, 1, H, W)
    """
    return F.binary_cross_entropy_with_logits(pred_seg, true_seg) + \
           F.l1_loss(pred_msk.flatten(0, 1), true_seg.flatten(0, 1)) + \
           laplacian_loss(pred_msk.flatten(0, 1), true_seg.flatten(0, 1))


# ----------------------------------------------------------------------------- Laplacian Loss


def laplacian_loss(pred, true, weight=None, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    loss = 0
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    if weight != None:
        weight_pyramid = normal_pyramid(weight, max_levels)
        for level in range(max_levels):
            loss += (2**level) * F.l1_loss(pred_pyramid[level] * weight_pyramid[level],
                                           true_pyramid[level] * weight_pyramid[level])
    else:
        for level in range(max_levels):
            loss += (2**level) * F.l1_loss(pred_pyramid[level],
                                           true_pyramid[level])
    return loss / max_levels


def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid


def normal_pyramid(img, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = current[:, :, ::2, ::2]
        pyramid.append(current)
        current = down
    return pyramid


def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor(
        [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6],
         [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]],
        device=device,
        dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel


def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img


def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img


def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out


def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]
