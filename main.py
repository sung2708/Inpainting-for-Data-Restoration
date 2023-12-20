from model.GAN import GAN
import torch
from torchvision import datasets, transforms
from PIL import Image

#get data from data folder
def data_loader():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    return train_loader

#train the model
def train():
    data_loader_ = data_loader()
    gan = GAN()
    gan.train(data_loader_)
    torch.save(gan.generator.state_dict(), 'generator.pkl')
    torch.save(gan.discriminator.state_dict(), 'discriminator.pkl')

#generate images
def generate():
    gan = GAN()
    gan.generator.load_state_dict(torch.load('generator.pkl'))
    fake_images = gan.generate(1)
    fake_images = fake_images.view(1, 28, 28)
    #save image
    img = transforms.ToPILImage()(fake_images)
    img.save('fake_images.png')
    

#main function
if __name__ == '__main__':
    train()
    generate()

