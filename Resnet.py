from __future__ import print_function # 맨 첫줄에 와야함
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import numpy as np
from matplotlib import pyplot as plt

def imshow(img):
    img = (img+1)/2
    img = img.squeeze()
    np_img = torch.from_numpy(img.numpy())
    plt.imshow(np_img, cmap='gray')
    plt.show()

def imshow_grid(img): 
    img = make_grid(img.cpu().detach())
    img = (img+1)/2
    npimg = torch.from_numpy(img.numpy())
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

devicename = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(devicename)
print(f'Training GAN Model on {device} with MNIST Data\n{"=" * 44}')


transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist_train = datasets.MNIST(root="MNIST/",
                             train=True,
                             download=True,
                             transform=transform)
mnist_test = datasets.MNIST(root="MNIST/",
                             train=True,
                             download=True,
                             transform=transform)

batch_size=100
train_loader = DataLoader(dataset=mnist_train,
                        batch_size=batch_size,
                        shuffle=True)
test_loader = DataLoader(dataset=mnist_test,
                       batch_size=batch_size,
                       shuffle=True)


class Generator(nn.Module): #fake data 생성(real dataset을 바탕으로)
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential( #deconvolutional
      nn.Linear(100, 256),
      nn.ReLU(),
      
      nn.Linear(256,512),
      nn.ReLU(),
      
      nn.Linear(512,1024),
      nn.ReLU(),
      nn.Linear(1024, 784), # 1024 = 32*32, 784 = 28*28
      nn.Tanh()
    )
  def forward(self, x):
    return self.main(x)

class Discriminator(nn.Module): #진짜같은 fake data를 구별
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential( #convolutional
      nn.Linear(784, 1024),
      nn.ReLU(),
      nn.Dropout(p=0.1),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Dropout(p=0.1),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(p=0.1),
      nn.Linear(256, 1),
      nn.Sigmoid()
    )
  def forward(self, x):
    return self.main(x)

generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer_g = optim.Adam(generator.parameters(), lr = 0.0002)
optimizer_d = optim.Adam(discriminator.parameters(), lr = 0.0002)

def sample_z(batch_size = 1, d_noise=100):
    return torch.randn(batch_size, d_noise, device=device)

def train(epoch):
  # tells your model that you are training the model.
  generator.train()
  discriminator.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)

    # maximize V(D,G) on D's sight
    p_real = discriminator(data.view(-1, 28*28)) # dimension을 모를 때 -1 사용
    p_fake = discriminator(generator(sample_z(batch_size, d_noise=100)))

    loss_real = -1 * torch.log(p_real)   # -1 for gradient ascending (log 미분과 관련)
    loss_fake = -1 * torch.log(1.-p_fake) # -1 for gradient ascending
    loss_d = (loss_real + loss_fake).mean()

    loss_d.backward()
    optimizer_d.step()

    # minimize V(D, G) on G's sight
    p_fake = discriminator(generator(sample_z(batch_size, d_noise=100)))
    loss_g = -1 * torch.log(p_fake).mean()
    loss_g.backward()
    
    optimizer_g.step()

    optimizer_g.zero_grad()
    optimizer_d.zero_grad()

def evaluate_model():
    p_real, p_fake = 0.,0. # 자료형 지정
    # tell that you are testing
    generator.eval()
    discriminator.eval()
        
    for img_batch, label_batch in test_loader:
        img_batch, label_batch = img_batch.to(device), label_batch.to(device) 
        
        with torch.autograd.no_grad():
            p_real += (torch.sum(discriminator(img_batch.view(-1, 28*28))).item())/10000.
            p_fake += (torch.sum(discriminator(generator(sample_z(batch_size, d_noise=100)))).item())/10000.
            
            
    return p_real, p_fake


p_real_trace = []
p_fake_trace = []

if __name__ == '__main__':
    for epoch in range(200):

        train(epoch)
        p_real, p_fake = evaluate_model()
        
        p_real_trace.append(p_real)
        p_fake_trace.append(p_fake) 
        if((epoch+1)% 25 == 0):
            print('(epoch %i/200) p_real: %f, p_g: %f' % (epoch+1, p_real, p_fake))
            imshow_grid(generator(sample_z(16)).view(-1, 1, 28, 28))
