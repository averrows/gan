from torch import nn
import torch
# size of z latent vector
nz = 100
# size of feature maps in generator
ngf = 64
nc = 3


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            # size of input: (nc) x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            # size of input: (ngf) x 32 x 32
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            # size of input: (ngf*2) x 16 x 16
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            # size of input: (ngf*4) x 8 x 8
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf * 8, 1, 4, 1, 0, bias=False),
            # size of input: (ngf*8) x 4 x 4
            nn.Sigmoid() # why sigmoid?
            # sigmoid is used to normalize the input between 0 and 1
        )
    
    def forward(self, input: torch.Tensor):
        return self.main(input)