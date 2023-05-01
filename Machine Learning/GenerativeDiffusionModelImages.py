import torchvision
import torch
from torchvision.transforms import Compose
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from diffusers import DDPMScheduler, UNet2DModel
import pickle


# Plots a 16x8 grid of images
def plotImages(x):
    plt.imshow(torchvision.utils.make_grid((x+1)/2, nrows=16)[0], cmap="Greys")
    plt.show()

# Calucatins the location x0 should be diffused to at time T
# x0 is the initial image
# alphabart is the alphaBar at time T
def ForwardDiffusionFast(x0, alphaBart):
    imageSize = x0.size()
    z = torch.tensor(st.norm.rvs(loc=0, scale=1, size=imageSize))
    xt = np.sqrt(alphaBart)*x0 + np.sqrt(1-alphaBart)*z
    return xt

# The Inverse of the ForwardDiffusionFast funtion to give z
# xt is the position at time t
# x0 is the initial positon
# alphaBar is the time-dependant alphaBar at time t
def guessZ(xt, x0, alphaBar):
    return (xt - (alphaBar)*x0)/((1 - alphaBar)**0.5)

# xt is the position of the particle at timestep t
# betas is a np array of time-dependant betas
# timesteps is number of timesteps
def undiffuse(xt, betas, timesteps, NeuralNet=None, image=None):
    # Defining alpha vectors
    alpha = 1 - betas
    alphaBar = np.cumprod(alpha)
    output = torch.zeros_like(xt)

    # Re-defining time array where 128 is batch size
    T = torch.from_numpy(np.array([timesteps-1 for i in range(0,128)]))
    T = T.reshape(128,1,1,1)

    x0 = image
    # Delta function Image
    #zgt = guessZ(xt, x0, alphaBar[T])
    # Neural Net
    #zgt = NeuralNet(xt, T.squeeze())
    # Two Delta functions
    #za = guessZ(locations[T], 0.4, alphaBarts[T])
    #zb = guessZ(locations[T], -0.6, alphaBarts[T])
    #zgt = ((0.8*za*np.exp(-(za**2)/2)) + (0.2*zb*np.exp(-(zb**2)/2)))/((0.8*np.exp(-(za**2)/2)) + (0.2*np.exp(-(zb**2)/2)))
    # Neural Network
    #zgt = NeuralNet(torch.tensor([float(locations[T]), T]))
    #zgt = NeuralNet(torch.tensor([float(locations[T]), T, -1]))
    
    # Defining terms
    sts = (1-alpha[T])/((1-alphaBar[T])**0.5)
    z = st.norm.rvs(loc=0, scale=1)
    
    betaTilde = (1 - alphaBar[T-1])/(1-alphaBar[T])*betas[T]
    if T[0, :, :, :] == 1:
        output = (xt - sts*zgt)/(alpha[T]**0.5)
    else:
        output = (xt - sts*zgt)/(alpha[T]**0.5) + (betaTilde**0.5)*z

    return output

def forwardDiffusionPlots(images):
    # Define betas array
    betaMax = 0.02
    timesteps = 200
    beta = torch.tensor(np.linspace(0.0001, betaMax, timesteps, dtype=np.float32))
    alpha = 1 - beta
    alphaBar = np.cumprod(alpha)

    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
    steps = [1, 50, 150, 199]
    ax[0].imshow(torchvision.utils.make_grid((images+1)/2, nrows=16)[0], cmap="Greys")
    ax[0].set_title("T = 0")
    for i in range(len(steps)):
        xt = ForwardDiffusionFast(images, alphaBar[steps[i]])
        ax[i+1].imshow(torchvision.utils.make_grid((xt+1)/2, nrows=16)[0], cmap="Greys")
        ax[i+1].set_title("T = {}".format(steps[i]))
    fig.suptitle("Fast Diffusion on MNIST Dataset")
    fig.savefig("Plots/MNISTFastDiffusion")
    plt.show()

def undiffuseImages():
    # Images
    image = next(iter(train_dataloader))[0]
    # Define betas array
    betaMax = 0.02
    timesteps = 200
    beta = torch.tensor(np.linspace(0.0001, betaMax, timesteps, dtype=np.float32))
    alpha = 1 - beta
    alphaBar = np.cumprod(alpha)
    # Time Array
    steps = [0, 1, 50, 150]

    # Forward diffuse to timestep 199
    x199 = ForwardDiffusionFast(image, alphaBar[199])
    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 5))
    ax[0].imshow(torchvision.utils.make_grid((x199+1)/2, nrows=16)[0], cmap="Greys")
    ax[0].set_title("T = 199")
    for i in range(len(steps)-1, -1, -1):
        xt = undiffuse(x199, beta, steps[i]+1, image=image)
        ax[len(steps)-i].imshow(torchvision.utils.make_grid((xt+1)/2, nrows=16)[0], cmap="Greys")
        ax[len(steps)-i].set_title("T = {}".format(steps[i]))
    fig.suptitle("Undiffusion on MNIST Dataset")
    fig.savefig("Plots/MNISTUndiffusion")
    plt.show()




if __name__ == "__main__":
    #Load MNIST dataset
    dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Lambda(lambda t: (t*2)-1)]))
    train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create the network
    NeuralNet = UNet2DModel(
        sample_size=28,  # the target image resolution
        in_channels=1,  # the number of input channels, 3 for RGB images
        out_channels=1,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64),  # Roughly matching our basic unet example
        down_block_types=( 
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",   # a regular ResNet upsampling block
        ),
    )

    # Use GPU
    

    

    # Define betas array
    #forwardDiffusionPlots(x)
    undiffuseImages()