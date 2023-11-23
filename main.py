from generator import Generator
from discriminator import Discriminator
from weight_init import weight_init
from torchsummary import summary

from torch import nn, optim
import torch

netG = Generator()
netG.apply(weight_init)
summary(netG, (100, 1, 1))
netD = Discriminator()
netD.apply(weight_init)
summary(netD, (3, 64, 64))

class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator.apply(weight_init)
        self.discriminator.apply(weight_init)
    def generate(self, input: torch.Tensor):
        return self.generator.forward(input)
    
    def discriminate(self, input: torch.Tensor):
        return self.discriminator.forward(input)
    
    def train(self, dataloader, num_epochs = 5):
        d = self.discriminator
        g = self.generator
        true_label = 1
        fake_label = 0
        criterion = nn.BCELoss()
        d_optimizer = optim.AdamW(d.parameters, lr=0.0002, betas=(0.5, 0.999))
        g_optimizer = optim.AdamW(g.parameters, lr=0.0002, betas=(0.5, 0.999))
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader):
                # train discriminator
                d.zero_grad()
                true_labels = torch.full((data.shape[0],), true_label)
                d_predict = d.forward(data)
                real_loss = criterion(d_predict, true_labels)
                real_loss.backward()
                
                # train generator
                noise = torch.randn(data.shape[0], 100, 1, 1)
                fake_images = g.forward(noise)
                fake_labels = torch.full((data.shape[0],), fake_label)
                d_predict = d.forward(fake_images)
                fake_loss = criterion(d_predict, fake_labels)
                fake_loss.backward()
                
                pass
        