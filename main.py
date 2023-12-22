from model.Discriminator import Discriminator
from model.Generator import Generator
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob
import os
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, img_size=128, mask_size=64, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob(f"{root}/*.jpg"))
        self.files = self.files[:-4000] if mode == "train" else self.files[-4000:]

    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked_part = img[:, y1:y2, x1:x2]
        masked_img = img.clone()
        masked_img[:, y1:y2, x1:x2] = 1

        return masked_img, masked_part

    def apply_center_mask(self, img):
        """Mask center part of image"""
        i = (self.img_size - self.mask_size) // 2
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, i

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.transform(img)
        if self.mode == "train":
            masked_img, aux = self.apply_random_mask(img)
        else:
            masked_img, aux = self.apply_center_mask(img)

        return img, masked_img, aux

    def __len__(self):
        return len(self.files)

dataset_name = "data/img_align_celeba"
batch_size = 64
n_cpu = 8
img_size = 128
channels = 3
mask_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 100
n_epochs = 15
sample_interval = 100
load_pretrained_models = False

cuda = torch.cuda.is_available()
os.makedirs(f"images/{dataset_name}", exist_ok=True)
os.makedirs(f"saved_models/{dataset_name}", exist_ok=True)

path_h = f"saved_models/{dataset_name}/Generator_0.pth"
path_g = f"saved_models/{dataset_name}/Discriminator_0.pth"

transforms_ = [
    transforms.Resize((img_size, img_size), Image.Resampling.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset(dataset_name, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_cpu,
)
test_dataloader = DataLoader(
    ImageDataset(dataset_name, transforms_=transforms_, mode="val"),
    batch_size=12,
    shuffle=True,
    num_workers=1,
)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm2d" in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def save_sample(batches_done):
    samples, masked_samples, i = next(iter(test_dataloader))
    samples = Variable(samples.type(Tensor))
    masked_samples = Variable(masked_samples.type(Tensor))
    i = i[0].item()
    gen_mask = generator(masked_samples)
    filled_samples = masked_samples.clone()
    filled_samples[:, :, i : i + mask_size, i : i + mask_size] = gen_mask
    sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
    save_image(sample, f"result/{batches_done}.png", nrow=6, normalize=True)

adversarial_loss = torch.nn.MSELoss()

generator = Generator(channels=channels)
discriminator = Discriminator(channels=channels)

if load_pretrained_models:
    generator.load_state_dict(torch.load(path_h))
    discriminator.load_state_dict(torch.load(path_g))

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

gen_adv_losses, gen_pixel_losses, disc_losses, counter = [], [], [], []
valid = Variable(Tensor(batch_size, 1, 1, 1).fill_(1.0), requires_grad=False)
fake = Variable(Tensor(batch_size, 1, 1, 1).fill_(0.0), requires_grad=False)
valid_expanded = valid.view(valid.size(0), valid.size(1), 1, 1).expand_as(discriminator(torch.empty(batch_size, channels, mask_size, mask_size)))

for epoch in range(n_epochs):
    gen_adv_loss, gen_pixel_loss, disc_loss = 0, 0, 0
    tqdm_bar = tqdm(dataloader, desc=f'Training Epoch {epoch} ', total=int(len(dataloader)))

    for i, (imgs, masked_imgs, masked_parts) in enumerate(tqdm_bar):
        imgs = Variable(imgs.type(Tensor))
        masked_imgs = Variable(masked_imgs.type(Tensor))
        masked_parts = Variable(masked_parts.type(Tensor))

        # Train Generator
        optimizer_G.zero_grad()
        gen_parts = generator(masked_imgs)

        # Adversarial and pixelwise loss
        g_adv = adversarial_loss(discriminator(gen_parts), valid_expanded)
        g_pixel = torch.mean(torch.abs(gen_parts - masked_parts))
        g_loss = 0.001 * g_adv + 0.999 * g_pixel
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real samples
        real_loss = adversarial_loss(discriminator(masked_parts), valid_expanded)

        # Ensure dimensions of 'fake' match the discriminator output after applying the generator's output
        fake_expanded = fake.view(fake.size(0), fake.size(1), 1, 1).expand_as(discriminator(generator(masked_imgs).detach()))

        # Measure discriminator's ability to classify fake samples
        fake_loss = adversarial_loss(discriminator(generator(masked_imgs).detach()), fake_expanded)

        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        optimizer_D.step()

        gen_adv_loss += g_adv.item()
        gen_pixel_loss += g_pixel.item()
        disc_loss += d_loss.item()
        counter.append(i)
        gen_adv_losses.append(g_adv.item())
        gen_pixel_losses.append(g_pixel.item())
        disc_losses.append(d_loss.item())
        tqdm_bar.set_postfix(
            gen_adv_loss=gen_adv_loss / (i + 1),
            gen_pixel_loss=gen_pixel_loss / (i + 1),
            disc_loss=disc_loss / (i + 1),
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            save_sample(batches_done)

    torch.save(generator.state_dict(), f"saved_models/{dataset_name}/Generator_{epoch}.pth")
    torch.save(discriminator.state_dict(), f"saved_models/{dataset_name}/Discriminator_{epoch}.pth")