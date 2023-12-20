import torch
from model.Discriminator import Discriminator
from model.Generator import Generator

class GAN():
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.loss = torch.nn.BCELoss()

    def train(self, data_loader):
        for epoch in range(100):
            for i, (real_images, _) in enumerate(data_loader):
                real_labels = torch.ones(real_images.size(0), 1)
                fake_labels = torch.zeros(real_images.size(0), 1)

                # Train the discriminator
                self.discriminator_optimizer.zero_grad()
                outputs = self.discriminator(real_images)
                real_loss = self.loss(outputs, real_labels)
                real_score = outputs

                noise = torch.randn(real_images.size(0), 100)
                fake_images = self.generator(noise)
                outputs = self.discriminator(fake_images)
                fake_loss = self.loss(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.discriminator_optimizer.step()

                # Train the generator
                self.generator_optimizer.zero_grad()
                noise = torch.randn(real_images.size(0), 100)
                fake_images = self.generator(noise)
                outputs = self.discriminator(fake_images)

                g_loss = self.loss(outputs, real_labels)
                g_loss.backward()
                self.generator_optimizer.step()

                if (i+1) % 100 == 0:
                    print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                          % (epoch, 100, i+1, 600, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()))

    def generate(self, num_images=1):
        noise = torch.randn(num_images, 100)
        fake_images = self.generator(noise)
        return fake_images
        