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
    loss[f'{tag}/pha_l1'] = F.l1_loss(pred_pha, true_pha)
    loss[f'{tag}/pha_coherence'] = F.mse_loss(pred_pha[:, 1:] - pred_pha[:, :-1],
                                              true_pha[:, 1:] - true_pha[:, :-1]) * 5
    loss[f'{tag}/pha_laplacian'] = laplacian_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1))
    loss[f'{tag}/pha_sobel'] = sobel_loss(pred_pha.flatten(0, 1), true_pha.flatten(0, 1)) * 5
    loss[f'{tag}/pha_kld'] = kld_loss(pred_pha, true_pha)
    # loss[f'{tag}/pha_bce'] = F.binary_cross_entropy_with_logits(pred_pha, true_pha) * 0.1
    # foreground losses
    true_msk = true_pha.gt(0)
    pred_fgr = pred_fgr * true_msk
    true_fgr = true_fgr * true_msk
    loss[f'{tag}/fgr_l1'] = F.l1_loss(pred_fgr, true_fgr)
    loss[f'{tag}/fgr_coherence'] = F.mse_loss(pred_fgr[:, 1:] - pred_fgr[:, :-1],
                                              true_fgr[:, 1:] - true_fgr[:, :-1]) * 5
    # Total
    loss[f'{tag}/total'] = loss[f'{tag}/pha_l1'] + loss[f'{tag}/pha_coherence'] \
                         + loss[f'{tag}/pha_laplacian'] + loss[f'{tag}/pha_sobel'] +  loss[f'{tag}/pha_bce'] \
                         + loss[f'{tag}/fgr_l1'] + loss[f'{tag}/fgr_coherence']

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

    loss['consistency/fgr'] = F.l1_loss(fgr_hat, fgr_bar) * 5
    loss['consistency/pha'] = F.l1_loss(pha_hat, pha_bar) * 5

    loss['consistency/total'] = loss['consistency/fgr'] + loss['consistency/pha']

    return loss


def segmentation_loss(pred_seg, true_seg):
    """
    Args:
        pred_seg: Shape(B, T, 1, H, W)
        true_seg: Shape(B, T, 1, H, W)
    """
    return F.binary_cross_entropy_with_logits(pred_seg, true_seg)



def sobel_loss(pred, true):
    kernel_v = torch.tensor(
        [[0, -1, 0],
         [0, 0, 0],
         [0, 1, 0]],
        device=pred.device,
        dtype=pred.dtype)
    kernel_h = torch.tensor(
        [[0, 0, 0],
         [-1, 0, 1],
         [0, 0, 0]],
        device=pred.device,
        dtype=pred.dtype)
    kernel_v = kernel_v[None, None, :, :]
    kernel_h = kernel_h[None, None, :, :]
    
    pred_v = F.conv2d(pred, kernel_v, padding=1)
    pred_h = F.conv2d(pred, kernel_h, padding=1)
    pred = torch.sqrt(torch.pow(pred_v, 2) + torch.pow(pred_h, 2) + 1e-6)

    true_v = F.conv2d(true, kernel_v, padding=1)
    true_h = F.conv2d(true, kernel_h, padding=1)
    true = torch.sqrt(torch.pow(true_v, 2) + torch.pow(true_h, 2) + 1e-6)

    return F.l1_loss(pred, true)


def kld_loss(q, p):
    loss = 0.0

    B, T = p.shape[:2]
    for b in range(B):
        for t in range(T):
            q_dist = q[b, t, 0] / torch.sum(q[b, t, 0])
            p_dist = p[b, t, 0] / torch.sum(p[b, t, 0])
            loss += F.kl_div((q_dist + 1e-6).log(), p_dist, reduction='mean')

    loss = loss / (B * T) * 1000
    return loss

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
    # B, C, H, W = img.shape
    # img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    # img = img.reshape(B, C, H, W)
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
