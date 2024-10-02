import torch
from torch import nn

from base import BaseNet



class Conv2dWithConstraint(nn.Conv2d):
    """Conv2D with weight normalization."""
    def __init__(self, *args, max_norm=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

# well let's first set up run. just do interactive run on ilcc. or set up csd
# now with these. what do i do. clear variable names, run well. ah the fuckin shapes right fuck


class EEGNet(BaseNet):
    """
    
    """
    def __init__(self, n_chans, n_classes, input_time_length, final_conv_length='auto', F1=8, D=2, F2=16, drop_prob=0.25):
        super().__init__()
        self.F1, self.D, self.F2 = F1, D, F2
        self.drop_prob = drop_prob

        self.ensure_4d = nn.Identity()  # Placeholder for any necessary dimension adjustment
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            Conv2dWithConstraint(F1, F1*D, (n_chans, 1), max_norm=1, bias=False, groups=F1),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(drop_prob)
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 16), groups=F1*D, padding=(0, 8), bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(drop_prob)
        )
        self.classify = nn.Sequential(
            nn.Conv2d(F2, n_classes, (1, final_conv_length if isinstance(final_conv_length, int) else input_time_length)),
            nn.LogSoftmax(dim=1)
        )

        self.initialize()

    def forward(self, x):
        x = self.ensure_4d(x)
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.classify(x)
        return x.squeeze()


# model = EEGNet(n_chans=64, n_classes=4, input_time_length=128, final_conv_length=32)
