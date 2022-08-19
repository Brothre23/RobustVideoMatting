
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampling_points import sampling_points, point_sample


class PointRendRefiner(nn.Module):
    def __init__(self, hidden_channels=16, beta=0.75):
        super().__init__()
        # self.mlp_fgr = nn.Conv1d(hidden_channels+3, 3, 1)
        # self.mlp_pha = nn.Conv1d(hidden_channels+1, 1, 1)
        self.mlp = nn.Conv1d(hidden_channels+4, 4, 1)
        # self.attent = nn.MultiheadAttention(embed_dim=20, num_heads=1, batch_first=True)
        self.beta = beta

    def forward_single_frame(self, src, hid, fgr, pha):
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """

        # frequency = torch.bincount((pha.flatten() * 255).int())
        # threshold = torch.argmin(frequency).float() / 255.0

        out = torch.cat([fgr, pha], dim=1)

        while out.shape[-2:] != src.shape[-2:]:
            out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=True)

            num_points = (out.shape[-1] // 8) * (out.shape[-2] // 8)
            points_idx, points = sampling_points(out, num_points, training=self.training)

            coarse = point_sample(out, points, align_corners=False)
            fine = point_sample(hid, points, align_corners=False)

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.mlp(feature_representation)

            B, C, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            out = (out.reshape(B, C, -1)
                      .scatter_(2, points_idx, rend)
                      .view(B, C, H, W))

        fgr, pha = out.split([3, 1], dim=1)

        return fgr, pha

    def forward_time_series(self, src, hid, fgr, pha):
        B, T = src.shape[:2]
        fgr, pha = self.forward_single_frame(
            src.flatten(0, 1),
            hid.flatten(0, 1),
            fgr.flatten(0, 1),
            pha.flatten(0, 1)
        )
        fgr = fgr.unflatten(0, (B, T))
        pha = pha.unflatten(0, (B, T))
        return fgr, pha

    def forward(self, src, hid, fgr, pha):
        if src.ndim == 5:
            return self.forward_time_series(src, hid, fgr, pha)
        else:
            return self.forward_single_frame(src, hid, fgr, pha)
