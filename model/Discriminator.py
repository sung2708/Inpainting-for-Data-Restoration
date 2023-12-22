import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(3, 128, kernel_size = 5, stride = 2, padding = 2, bias = False),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(128, 256, kernel_size = 5, stride = 2, padding = 2, bias = False),
      nn.BatchNorm2d(256),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(256, 512, kernel_size = 5, stride = 2, padding =2, bias = False),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(512, 1024, kernel_size = 5, stride = 2, padding = 2, bias = False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(1024, 1, kernel_size = 4, stride = 1, padding = 0, bias = False)
    )
    
  def forward(self, x):
    x = self.main(x)
    return x