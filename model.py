"""
A small UNet for image segmentation.

A UNet has two halves:
  1. The ENCODER shrinks the image step by step, learning "what" is in it.
  2. The DECODER grows it back to full size, working out "where" things are.

"Skip connections" pass detail from each encoder step straight across to the
matching decoder step, so fine edges aren't lost while the image is small.

Instead of max pooling, this model shrinks the image with STRIDED CONVOLUTIONS:
a convolution that moves 2 pixels at a time, so the output is half the size.
Unlike max pooling, the network gets to *learn* how to downsample.

Image sizes for a 128x128 input:

    128x128  ->  64x64  ->  32x32  ->  16x16     (encoder)
    16x16    ->  32x32  ->  64x64  ->  128x128   (decoder)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    The basic building block: two 3x3 convolutions in a row.
    Each one is followed by BatchNorm (keeps training stable) and ReLU (adds non-linearity).
    padding=1 keeps the height and width the same.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class DownBlock(nn.Module):
    """
    One encoder step: halve the image size with a strided convolution,
    then process it with a DoubleConv.

    stride=2 means the filter jumps 2 pixels at a time, so 128x128 -> 64x64.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """
    One decoder step: double the image size with a transposed convolution
    (the opposite of a strided convolution), stick the matching encoder output
    onto it (the skip connection), then process it with a DoubleConv.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)  # join along the channel dimension
        x = self.conv(x)
        return x


class SmallUNet(nn.Module):
    """
    The full model.

    Input:  a batch of RGB images, shape (batch, 3, H, W)
    Output: one score per pixel, shape (batch, 1, H, W)
            A positive score means "pet", a negative score means "background".

    H and W must be divisible by 8, because the image is halved 3 times.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder: channels go up while the image gets smaller
        self.start = DoubleConv(in_channels, 16)  # 128x128, 16 channels
        self.down1 = DownBlock(16, 32)            #  64x64,  32 channels
        self.down2 = DownBlock(32, 64)            #  32x32,  64 channels
        self.down3 = DownBlock(64, 128)           #  16x16, 128 channels (the "bottleneck")

        # Decoder: channels go down while the image gets bigger
        self.up1 = UpBlock(128, 64, 64)           #  32x32
        self.up2 = UpBlock(64, 32, 32)            #  64x64
        self.up3 = UpBlock(32, 16, 16)            # 128x128

        # A 1x1 convolution turns the 16 channels into 1 score per pixel
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder: keep each output so the decoder can use it later
        s1 = self.start(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        bottleneck = self.down3(s3)

        # Decoder: grow back up, joining on the matching encoder outputs
        x = self.up1(bottleneck, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)

        return self.final(x)


# Run `python model.py` to check that the model works on a fake batch
if __name__ == "__main__":
    model = SmallUNet()
    fake_images = torch.randn(2, 3, 128, 128)
    output = model(fake_images)
    num_params = sum(p.numel() for p in model.parameters())
    print("Input shape: ", tuple(fake_images.shape))
    print("Output shape:", tuple(output.shape))
    print(f"Parameters:   {num_params:,}")
