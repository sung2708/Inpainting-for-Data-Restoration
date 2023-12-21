from re import S
import torch

class GAN():
    def __init__(self, generator, discriminator, device):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.loss = torch.nn.BCELoss()
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    
    def train(self, dataloader, epochs, batch_size, sample_interval):
        for epoch in range(epochs):
            for i, imgs in enumerate(dataloader):
                imgs = imgs.to(self.device)
                real = torch.ones((imgs.size(0), 1)).to(self.device)
                fake = torch.zeros((imgs.size(0), 1)).to(self.device)

                # train generator
                self.generator_optimizer.zero_grad()
                z = torch.randn((imgs.size(0), 100)).to(self.device)
                gen_imgs = self.generator(z)
                g_loss = self.loss(self.discriminator(gen_imgs), real)
                g_loss.backward()

                # train discriminator
                self.discriminator_optimizer.zero_grad()
                real_loss = self.loss(self.discriminator(imgs), real)
                fake_loss = self.loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.discriminator_optimizer.step()

                if i % sample_interval == 0:
                    print('Epoch: {}, Batch: {}, D Loss: {}, G Loss: {}'.format(epoch, i, d_loss.item(), g_loss.item()))

    def generate(self, z):
        return self.generator(z)
    
