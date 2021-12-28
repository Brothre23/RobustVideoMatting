import torch
from torch import nn
from .MicroNet.model import MicroNet
from torchvision.transforms.functional import normalize

class MicroNetEncoder(MicroNet):
    def __init__(self, pretrained: bool = False):
        super().__init__(input_size=256)

        if pretrained:
            import torch
            self.load_state_dict(torch.load('model/MicroNet/micronet-m0.pth'))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        del self.classifier

    def forward_single_frame(self, x):
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        x = self.upsample(x)

        x = self.features[0](x)
        x = self.features[1](x)
        f1 = x

        x = self.features[2](x)
        f2 = x

        x = self.features[3](x)
        x = self.features[4](x)
        f3 = x

        x = self.features[5](x)
        x = self.features[6](x)
        x = self.features[7](x)
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
