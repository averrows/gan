from torch import nn


# size of z latent vector
nz = 100
# size of feature maps in generator
ngf = 64

nc = 3

# what is the difference between conv and convtranspose?
# convtranspose is used for upsampling
# conv is used for downsampling
# upsa

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), # what is batchnorm?
            # batchnorm is used to normalize the input layer by re-centering and re-scaling
            nn.ReLU(True),
            #inplace = True means that it will modify the input directly, without allocating any additional output
            
            # size of input: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # size of input: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), # why is padding 1?
            # padding is 1 because we want to keep the size of the image the same
            # if no padding, then the size of the image will be reduced by 1
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # size of input: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), # why is padding 1?
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # size of input: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), # why is padding 1?
            nn.Tanh() # why tanh?
            # tanh is used to normalize the input between -1 and 1
            # tanh is used because the input is normalized between -1 and 1
            # because the output we want is between -1 and 1

            # size of input: (nc) x 64 x 64
            
        )

    def forward(self, input):
        return self.main(input)