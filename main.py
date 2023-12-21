from model.Discriminator import Discriminator
from model.Generator import Generator
from model.GAN import GAN
import torch
import os
import PIL.Image as Image
import torchvision.transforms as transforms
import torchvision.utils

#define data path
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = os.listdir(root)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.imgs[index])
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(os.listdir(self.root))
    
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset_name = 'data/img_align_celeba'
dataloader = torch.utils.data.DataLoader(ImageDataset(dataset_name, transform=transform), batch_size=64, shuffle=True)


# define model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator().to(device)
D = Discriminator().to(device)
gan = GAN(G, D, device)

# train model
gan.train(dataloader, epochs=5, batch_size=64, sample_interval=400)

#save model
torch.save(G.state_dict(), 'G.pth')
torch.save(D.state_dict(), 'D.pth')

#load model
G = Generator().to(device)
D = Discriminator().to(device)
G.load_state_dict(torch.load('G.pth'))
D.load_state_dict(torch.load('D.pth'))


# inpainting image from input folder
for i in range(10):
    img = Image.open('input/{}.jpg'.format(i)).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    z = torch.randn(1, 100, 1, 1).to(device)
    fake_img = G(z)
    fake_img = fake_img.cpu().data
    torchvision.utils.save_image(fake_img, 'output/{}.jpg'.format(i), normalize=True)

