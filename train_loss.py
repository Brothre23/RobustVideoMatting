import torch
from torch.nn import functional as F
from einops import repeat

# --------------------------------------------------------------------------------- Train Loss


def matting_loss(pred_fgr, pred_pha, true_fgr, true_pha, tag):
    """
    Args:
        pred_fgr: Shape(B, T, 3, H, W)
        pred_pha: Shape(B, T, 1, H, W)
        true_fgr: Shape(B, T, 3, H, W)
        true_pha: Shape(B, T, 1, H, W)
    """
    loss = dict()
    # alpha losses
    loss[f'{tag}/pha_l2'] = F.mse_loss(pred_pha, true_pha)
    loss[f'{tag}/pha_laplacian'] = laplacian_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1))
    loss[f'{tag}/pha_coherence'] = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1], 
                                              true_pha[:, 1:] - true_pha[:, :-1]) * 5
    # foreground losses
    true_msk = true_pha.gt(0)
    pred_fgr = pred_fgr * true_msk
    true_fgr = true_fgr * true_msk
    loss[f'{tag}/fgr_l2'] = F.mse_loss(pred_fgr, true_fgr)
    loss[f'{tag}/fgr_laplacian'] = laplacian_loss(pred_fgr.flatten(0, 1), true_fgr.flatten(0, 1))
    loss[f'{tag}/fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
                                       true_fgr[:, 1:] - true_fgr[:, :-1]) * 5
    # Total
    loss[f'{tag}/total'] = loss[f'{tag}/pha_l2'] + loss[f'{tag}/pha_coherence'] + loss[f'{tag}/pha_laplacian'] \
                         + loss[f'{tag}/fgr_l2'] + loss[f'{tag}/fgr_coherence'] + loss[f'{tag}/fgr_laplacian']

    return loss


def consistency_loss(fgr_hat, pha_hat, fgr_bar, pha_bar):
    """
    Args:
        fgr_hat: Shape(B, T, 3, H, W)
        pha_hat: Shape(B, T, 1, H, W)
        fgr_bar: Shape(B, T, 3, H, W)
        pha_bar: Shape(B, T, 1, H, W)
    """
    loss = dict()
    
    # loss['consistency/fgr_l2'] = F.mse_loss(fgr_hat, fgr_bar) * 2
    # loss['consistency/pha_l2'] = F.mse_loss(pha_hat, pha_bar) * 2

    # loss['consistency/fgr_coherence'] = F.mse_loss(fgr_hat[:, 1:] - fgr_hat[:, :-1],
    #                                         fgr_bar[:, 1:] - fgr_bar[:, :-1]) * 10
    # loss['consistency/pha_coherence'] = F.mse_loss(pha_hat[:, 1:] - pha_hat[:, :-1],
    #                                         pha_bar[:, 1:] - pha_bar[:, :-1]) * 10

    # loss['consistency/total'] = loss['consistency/fgr_l2'] + loss['consistency/pha_l2'] + \
    #                             loss['consistency/fgr_coherence'] + loss['consistency/pha_coherence']
    loss['consistency/fgr'] = F.mse_loss(fgr_hat, fgr_bar) * 5
    loss['consistency/pha'] = F.mse_loss(pha_hat, pha_bar) * 5

    loss['consistency/total'] = loss['consistency/fgr'] + loss['consistency/pha']

    return loss


def segmentation_loss(pred_seg, true_seg):
    """
    Args:
        pred_seg: Shape(B, T, 1, H, W)
        true_seg: Shape(B, T, 1, H, W)
    """
    return F.binary_cross_entropy_with_logits(pred_seg, true_seg)


# def optical_flow_loss(pred, true, size, op_model, channel):
#     pred_op_list = []
#     true_op_list = []

#     for i in range(size - 1):
#         padder = InputPadder(pred[:, i, :, :, :].shape)

#         image1, image2 = pred[:, i, :, :, :], pred[:, i, :, :, :]
#         if channel == 1:
#             image1 = repeat(image1, 'b c h w -> b (repeat c) h w', repeat=3)
#             image2 = repeat(image2, 'b c h w -> b (repeat c) h w', repeat=3)
#         image1, image2 = padder.pad(image1, image2)
#         _, result = op_model(image1, image2, iters=10)
#         pred_op_list.append(result)

#         image1, image2 = true[:, i, :, :, :], true[:, i, :, :, :]
#         if channel == 1:
#             image1 = repeat(image1, 'b c h w -> b (repeat c) h w', repeat=3)
#             image2 = repeat(image2, 'b c h w -> b (repeat c) h w', repeat=3)
#         image1, image2 = padder.pad(image1, image2)
#         _, result = op_model(image1, image2, iters=10)
#         true_op_list.append(result)

#     pred_op_tensor = torch.stack(pred_op_list).permute(1, 0, 2, 3, 4)
#     true_op_tensor = torch.stack(true_op_list).permute(1, 0, 2, 3, 4)

#     return F.mse_loss(pred_op_tensor, true_op_tensor)


# ----------------------------------------------------------------------------- Laplacian Loss


def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
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
