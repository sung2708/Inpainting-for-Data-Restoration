from model.Discriminator import Discriminator
from model.Generator import Generator
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import torch
import glob

folder_data = glob.glob("./data/img_align_celeba/*.jpg")
len_data = len(folder_data)

train_image_paths = folder_data[0:200000]

class TrainDataset(Dataset):
    def __init__(self, image_paths, train=True):
        self.image_paths = image_paths
        self.transforms = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        t_image = self.transforms(image)
        return t_image

    def __len__(self):
        return len(self.image_paths)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

train_dataset = TrainDataset(train_image_paths)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

cuda = False
Tensor = torch.FloatTensor  # Use CPU Tensor

generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

def noise(size):
    n = Variable(torch.randn(size, 100, 1, 1))
    n = n.to(cuda) if cuda else n
    return n

samples = 16
fixed_noise = noise(samples)

lr = 0.0002
b1 = 0.5
b2 = 0.999

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

def train():
    for epoch in range(15):
        for i, imgs in tqdm(enumerate(train_loader)):
            imgs = imgs.to(cuda) if cuda else imgs
            batch_size = imgs.shape[0]

            valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

            real_imgs = Variable(imgs.type(Tensor))

            optimizer_G.zero_grad()

            z = noise(batch_size)
            gen_imgs = generator(z)

            g_loss = torch.mean((discriminator(gen_imgs) - valid) ** 2)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            real_loss = torch.mean((discriminator(real_imgs) - valid) ** 2)
            fake_loss = torch.mean((discriminator(gen_imgs.detach()) - fake) ** 2)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, 200, i, len(train_loader), d_loss.item(), g_loss.item())
                )

        save_image(gen_imgs.data[:25], "images/%d.png" % epoch, nrow=5, normalize=True)

    # Save model after training
    torch.save(generator.state_dict(), './generator.pth')
    torch.save(discriminator.state_dict(), './discriminator.pth')

if __name__ == "__main__":
    train()
