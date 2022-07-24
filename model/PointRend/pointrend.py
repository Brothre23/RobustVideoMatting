
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import kornia

from .sampling_points import sampling_points, point_sample


class PointRendRefiner(nn.Module):
    def __init__(self, hidden_channels=16, beta=0.75):
        super().__init__()
        # self.mlp_fgr = nn.Conv1d(hidden_channels+3, 3, 1)
        # self.mlp_pha = nn.Conv1d(hidden_channels+1, 1, 1)
        self.mlp = nn.Conv1d(hidden_channels+4, 4, 1)
        # self.attent = nn.MultiheadAttention(embed_dim=20, num_heads=1, batch_first=True)
        # self.kernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) * 1.0eee
        self.beta = beta

    def forward_single_frame(self, src, hid, fgr, pha, err):
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """

        # while pha.shape[-2:] != src.shape[-2:]:
        #     fgr = F.interpolate(fgr, scale_factor=2, mode="bilinear", align_corners=True)
        #     pha = F.interpolate(pha, scale_factor=2, mode="bilinear", align_corners=True)

        #     num_points = (pha.shape[-1] // 8) * (pha.shape[-2] // 8)

        #     points_idx, points = sampling_points(pha, num_points, training=self.training)

        #     coarse_fgr = point_sample(fgr, points, align_corners=False)
        #     coarse_pha = point_sample(pha, points, align_corners=False)
        #     fine = point_sample(hid, points, align_corners=False)

        #     feature_representation_fgr = torch.cat([coarse_fgr, fine], dim=1)
        #     feature_representation_pha = torch.cat([coarse_pha, fine], dim=1)

        #     rend_fgr = self.mlp_fgr(feature_representation_fgr)
        #     rend_pha = self.mlp_pha(feature_representation_pha)

        #     B, C, H, W = fgr.shape
        #     points_idx_expand = points_idx.unsqueeze(1).expand(-1, C, -1)
        #     fgr = (fgr.reshape(B, C, -1)
        #               .scatter_(2, points_idx_expand, rend_fgr)
        #               .view(B, C, H, W))

        #     B, C, H, W = pha.shape
        #     points_idx_expand = points_idx.unsqueeze(1).expand(-1, C, -1)
        #     pha = (pha.reshape(B, C, -1)
        #               .scatter_(2, points_idx_expand, rend_pha)
        #               .view(B, C, H, W))

        out = torch.cat([fgr, pha], dim=1)
        kernel = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).to(out.device) * 1.0

        while out.shape[-2:] != src.shape[-2:]:
            out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=True)
            err = F.interpolate(err, scale_factor=2.0, mode="bilinear", align_corners=True)

            num_points = (out.shape[-1] // 8) * (out.shape[-2] // 8)
            points_idx, points = sampling_points(out, err, kernel, num_points, training=self.training)

            coarse = point_sample(out, points, align_corners=False)
            fine = point_sample(hid, points, align_corners=False)

            feature_representation = torch.cat([coarse, fine], dim=1)

            # feature_representation = torch.transpose(feature_representation, 1, 2)
            # attented_feature, _ = self.attent(feature_representation, feature_representation, feature_representation)
            # attented_feature = torch.transpose(attented_feature, 1, 2)

            rend = self.mlp(feature_representation)

            B, C, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            out = (out.reshape(B, C, -1)
                      .scatter_(2, points_idx, rend)
                      .view(B, C, H, W))

        fgr, pha = out.split([3, 1], dim=1)

        return fgr, pha

    def forward_time_series(self, src, hid, fgr, pha, err):
        B, T = src.shape[:2]
        fgr, pha = self.forward_single_frame(
            src.flatten(0, 1),
            hid.flatten(0, 1),
            fgr.flatten(0, 1),
            pha.flatten(0, 1),
            err.flatten(0, 1)
        )
        fgr = fgr.unflatten(0, (B, T))
        pha = pha.unflatten(0, (B, T))
        return fgr, pha

    def forward(self, src, hid, fgr, pha, err):
        if src.ndim == 5:
            return self.forward_time_series(src, hid, fgr, pha, err)
        else:
            return self.forward_single_frame(src, hid, fgr, pha, err)

    # def forward(self, x, res2, out):
    #     """
    #     1. Fine-grained features are interpolated from res2 for DeeplabV3
    #     2. During training we sample as many points as there are on a stride 16 feature map of the input
    #     3. To measure prediction uncertainty
    #        we use the same strategy during training and inference: the difference between the most
    #        confident and second most confident class probabilities.
    #     """
    #     if not self.training:
    #         return self.inference(x, res2, out)

    #     points = sampling_points(out, x.shape[-1] // 16, self.k, self.beta)

    #     coarse = point_sample(out, points, align_corners=False)
    #     fine = point_sample(res2, points, align_corners=False)

    #     feature_representation = torch.cat([coarse, fine], dim=1)

    #     rend = self.mlp(feature_representation)

    #     return {"rend": rend, "points": points}

    # @torch.no_grad()
    # def inference(self, x, res2, out):
    #     """
    #     During inference, subdivision uses N=8096
    #     (i.e., the number of points in the stride 16 map of a 1024×2048 image)
    #     """
    #     num_points = 8096

    #     while out.shape[-1] != x.shape[-1]:
    #         out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

    #         points_idx, points = sampling_points(out, num_points, training=self.training)

    #         coarse = point_sample(out, points, align_corners=False)
    #         fine = point_sample(res2, points, align_corners=False)

    #         feature_representation = torch.cat([coarse, fine], dim=1)

    #         rend = self.mlp(feature_representation)

    #         B, C, H, W = out.shape
    #         points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
    #         out = (out.reshape(B, C, -1)
    #                   .scatter_(2, points_idx, rend)
    #                   .view(B, C, H, W))

    #     return {"fine": out}
