from torch import nn

class UNet(nn.Module):
    def __init__(self, in_channels, num_segment_classes):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_segment_classes, kernel_size=2, stride=2, output_padding=0)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        # Middle
        x2 = self.middle(x1)

        # Decoder
        x3 = self.decoder(x2)

        # Resize the output to match the original input size
        output_size = x.size()[2:]
        x3_resized = nn.functional.interpolate(x3, size=output_size, mode='bilinear', align_corners=False)

        return x3_resized