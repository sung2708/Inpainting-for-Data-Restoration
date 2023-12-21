import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, channels=3):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, kernel_size=3, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers


        def upsample(in_feat, out_feat, kernel_size=4, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, kernel_size, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *downsample(100, 64, kernel_size=3, normalize=False),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),
            nn.Conv2d(512, 4000, 1),
            *upsample(4000, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),
            nn.Conv2d(64, channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z.view(z.size(0), -1, 1, 1))
        return img
