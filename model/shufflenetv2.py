import torch
from torch import nn
from torchvision.models.shufflenetv2 import ShuffleNetV2
from torchvision.transforms.functional import normalize

class ShuffleNetV2Encoder(ShuffleNetV2):
    def __init__(self, pretrained: bool = False):
        super().__init__(stages_repeats=[4, 8, 4],
        stages_out_channels=[24, 48, 96, 192, 1024])

        if pretrained:
            import torch
            self.load_state_dict(torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth'))

        del self.maxpool
        del self.fc

    def forward_single_frame(self, x):
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        x = self.conv1(x)
        f1 = x
        x = self.stage2(x)
        f2 = x
        x = self.stage3(x)
        f3 = x
        x = self.stage4(x)
        x = self.conv5(x)
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
