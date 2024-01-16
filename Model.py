import os 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split

import torch.nn
import torchvision
import torch.nn as nn
from torchvision import transforms ,datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

from tqdm.notebook import tqdm

#Hyperparameter
device = "cuda" if torch.cuda.is_available() else "cpu"
lr=3e-4
z_dim=64
img_dim=28*28*1 #MNIST Dataset
batch_size=32
num_epochs=500



class Discrimator(nn.Module):
    def __init__(self,img_dim):
        super().__init__()
        self.disc=nn.Sequential(
            nn.Linear(img_dim,128),
            nn.LeakyReLU(0.1),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
        

    def forward(self,Dataset):
        X=self.disc(Dataset)
        return(X)

class Generator(nn.Module):
    def __init__(self,z_dim,img_dim):
        super().__init__()
        self.gen=nn.Sequential(
            nn.Linear(z_dim,256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,img_dim),
            nn.Tanh()
        )

    def forward(self,Dataset):

        return(self.gen(Dataset))



#Init the models

disc=Discrimator(img_dim).to(device)
gen=Generator(z_dim,img_dim).to(device)
Fixed_noise=torch.randn((batch_size,z_dim)).to(device)

Transforms=transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]
)


#DataSet loading
dataset=datasets.MNIST(root="/home/yash/CNN_Data",download=True,transform=Transforms)
loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

#Optimizer
opt_disc=optim.Adam(disc.parameters(),lr=lr)
opt_Gen=optim.Adam(gen.parameters(),lr=lr)

criterion=nn.BCELoss()

#Tensorboard Summary Runner

writer_fake=SummaryWriter(f"run/Gan_MNSIT/Fake")
writer_Real=SummaryWriter(f"run/Gan_MNSIT/Real")
step=0

for epoch in range(num_epochs):
    for batch_idx, (real,_) in enumerate(loader):
        real=real.view(-1,784).to(device)
        batch_size=real.shape[0]


        ## Training the Discrimator

        noise=torch.randn((batch_size,z_dim)).to(device)
        fake=gen(noise)
        disc_real=disc(real).view(-1)
        
        ##Loss Function Creation

        lossD_real=criterion(disc_real,torch.ones_like(disc_real))
        disc_fake=disc(fake).view(-1)
        lossD_Fake=criterion(disc_fake,torch.zeros_like(disc_fake))
        
        lossD=(lossD_real+lossD_Fake)/2

        disc.zero_grad()
        lossD.backward(retain_graph= True)
        opt_disc.step()


        ##Training the Generator
        output=disc(fake)
        LossG=criterion(output,torch.ones_like(output)) 
        gen.zero_grad()
        LossG.backward()
        opt_Gen.step()


        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {LossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(Fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_Real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

