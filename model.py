import torch.nn as nn


class Conv3dAutoencoder(nn.Module, latent):
    def __init__(self):
        super(Conv3dAutoencoder, self).__init__()
        latent = 1000

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # output=(6, 56, 56)
            nn.ReLU(True),
            nn.Conv3d(64, 32, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1)),  # output=(3, 28, 28)
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32 * 3 * 28 * 28, latent),  # Adjust the size based on the flattened output
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent, 32 * 3 * 28 * 28),
            nn.Unflatten(1, (32, 3, 28, 28)),  # Unflatten back to (32, 5, 25, 25)
            nn.ConvTranspose3d(
                32, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(0, 1, 1)
            ),  # output=(8, 56, 56)
            nn.ReLU(True),
            nn.ConvTranspose3d(
                64, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
            ),  # output=(10, 111, 111)
            nn.ReLU(True),
            nn.ConvTranspose3d(
                32, 1, kernel_size=(3, 2, 2), stride=(1, 1, 1), padding=(0, 0, 0)
            ),  # output=(12, 112, 112)
            nn.Sigmoid(),  # to scale output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
