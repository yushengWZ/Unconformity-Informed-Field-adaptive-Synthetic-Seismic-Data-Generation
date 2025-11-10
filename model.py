import torch
from torch import nn
import torch.nn.functional as F


# ***************************************************** CycleGAN *************************************************** #
class Generator(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False, batch_norm=False, up_mode='upconv'):
        super(Generator, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(input_dims, hidden_dims, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(hidden_dims),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dims, hidden_dims * 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(hidden_dims * 4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dims * 4, hidden_dims * 8, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(hidden_dims * 8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dims * 8, hidden_dims * 4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(hidden_dims * 4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dims * 4, hidden_dims * 1, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(hidden_dims * 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dims * 1, output_dims, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, input_dims):
        """Forward the discriminator."""
        out = self.layer(input_dims)
        return out


# ***************************************************** CoGAN *************************************************** #
class CoupledGenerators(nn.Module):
    def __init__(self, channels, img_size, latent_dim):
        super(CoupledGenerators, self).__init__()

        self.init_size = img_size // 4
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.shared_conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.G1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Tanh(),

            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.G2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Tanh(),

            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.fc(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img_emb = self.shared_conv(out)
        img1 = self.G1(img_emb)
        img2 = self.G2(img_emb)
        return img1, img2


class CoupledDiscriminators(nn.Module):
    def __init__(self, channels, img_size):
        super(CoupledDiscriminators, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block

        self.shared_conv = nn.Sequential(
            *discriminator_block(channels, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
        )
        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.D1 = nn.Linear(256 * ds_size ** 2, 1)
        self.D2 = nn.Linear(256 * ds_size ** 2, 1)

    def forward(self, img1, img2):
        # Determine validity of first image
        out = self.shared_conv(img1)
        out = out.view(out.shape[0], -1)
        validity1 = self.D1(out)
        # Determine validity of second image
        out = self.shared_conv(img2)
        out = out.view(out.shape[0], -1)
        validity2 = self.D2(out)

        return validity1, validity2


# ***************************************************** BiGAN *************************************************** #
class BiGenerator(nn.Module):

    def __init__(self):
        super(BiGenerator, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 1, kernel_size=3, padding=1, stride=1),
        )

    def forward(self, input_dims):
        out = self.layer(input_dims)
        return out


class BiDiscriminator(nn.Module):

    def __init__(self):
        super(BiDiscriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Flatten(),
            nn.Linear(65536, 1),
        )

    def forward(self, inputA, inputB):
        input = torch.cat([inputA, inputB], 1)
        out = self.layer(input)
        return out
