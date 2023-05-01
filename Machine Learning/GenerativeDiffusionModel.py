import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import torch
import torchvision
from torch import nn
import pickle


# Applies the Langevin Equation and returns the final location
# x_0 is the initial position
# T is the total number of steps
# delta is factor for the randomess
def ForwardDiffusion(x_0, T, delta):
    locations = [x_0]
    xOld = x_0
    for i in range(T):
        # Energy function is a simple harmonic oscillator
        Energy = (xOld**2)/2
        # Drawing from a gaussian centered on 0 with 1 standard deviation
        z = st.norm.rvs(loc=0, scale=1)
        # Langevin Equation
        xNew = xOld +(delta/2)*(-xOld) + np.sqrt(delta)*z
        locations.append(xNew)
        xOld = xNew
    return locations

# Does the Langevin Dynamcis but this time the beta is a function of time
# The output is the final location according to the Langevin Dynamics
# x_0 is the starting position
# T is the number of timesteps
# betas is the numpy array of timestep dependant beta values
def ForwardDiffusion2(x_0, T, betas):
    xNew = x_0
    locations = [x_0]
    for i in range(T):
        # Drawing from a gaussian centered on 0 with 1 standard deviation
        z = st.norm.rvs(loc=0, scale=1)
        # Langevin Equation
        xNew = xNew*((1-betas[i])**(0.5)) + z*(betas[i]**(0.5))
        locations.append(xNew)
    return xNew, locations

# Does Foward Diffusion but faster by skipping the calculations in the middle
# x_0 is the initial position
# alphaBart is a numpy array of alpha bars
# timeSteps is the number of Steps
def ForwardDiffusionFast(x_0, alphaBart):
    z = st.norm.rvs(loc=0, scale=1, size=len(alphaBart))
    xlocations = np.sqrt(alphaBart)*x_0 + np.sqrt(1-alphaBart)*z
    return xlocations

# Same as the forwrd diffusion fast function but also returns noise for training
def ForwardDiffusionFastWithNoise(x_0, alphaBart):
    z = st.norm.rvs(loc=0, scale=1, size=len(alphaBart))
    xlocations = np.sqrt(alphaBart)*x_0 + np.sqrt(1-alphaBart)*z
    return z[-1], xlocations

# The Inverse of the ForwardDiffusionFast funtion to give z
# xt is the position at time t
# x0 is the initial positon
# alphaBar is the time-dependant alphaBar at time t
def guessZ(xt, x0, alphaBar):
    return (xt - (alphaBar)*x0)/((1 - alphaBar)**0.5)

# Returns x0 for undiffuse
def sampleFromPInit():
    #return 0.4
    return np.random.choice([0.4, -0.6], p=[0.8, 0.2])

# xt is the position of the particle at timestep t
# betas is a np array of time-dependant betas
# timesteps is number of timesteps
def undiffuse(xt, betas, timesteps, NeuralNet=None):
    # Defining alpha vectors
    alphats = 1 - betas
    alphaBarts = np.cumprod(alphats)
    locations = np.zeros(timesteps)
    locations[timesteps-1] = xt

    for T in range(timesteps-1, 0, -1):
        x0 = sampleFromPInit()
        # Delta function
        #zgt = guessZ(locations[T], 0.4, alphaBarts[T])
        # Two Delta functions
        #za = guessZ(locations[T], 0.4, alphaBarts[T])
        #zb = guessZ(locations[T], -0.6, alphaBarts[T])
        #zgt = ((0.8*za*np.exp(-(za**2)/2)) + (0.2*zb*np.exp(-(zb**2)/2)))/((0.8*np.exp(-(za**2)/2)) + (0.2*np.exp(-(zb**2)/2)))
        # Neural Network
        zgt = NeuralNet(torch.tensor([float(locations[T]), T]))
        #zgt = NeuralNet(torch.tensor([float(locations[T]), T, -1]))
        
        # Defining terms
        sts = (1-alphats[T])/((1-alphaBarts[T])**0.5)
        z = st.norm.rvs(loc=0, scale=1)
        
        betaTilde = (1 - alphaBarts[T-1])/(1-alphaBarts[T])*betas[T]
        if T == 1:
            locations[T-1] = (locations[T] - sts*zgt)/(alphats[T]**0.5)
        else:
            locations[T-1] = (locations[T] - sts*zgt)/(alphats[T]**0.5) + (betaTilde**0.5)*z

    return locations

# Plotting the different approachs to visualize diffusion
def diffusionGraphs():
    # Approach 1
    locations1 = ForwardDiffusion(0, 100000, 0.01)
    # xVals for theory
    xVals = np.linspace(-3, 3, 1001)
    #Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(locations1[1000:], bins=100, density=True, label="Experimental Result")
    ax.plot(xVals, np.exp((-xVals**2)/2)/np.sqrt(2*np.pi), zorder=100, label="Theorectical Distribution")
    ax.legend(loc="upper left")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Location")
    fig.suptitle("Probability distribution of locations \n using Langevin dynamics")
    #fig.savefig("Plots/Fast Diffusion Approach 1")

    # Approach 2 SHO Energy
    delta = 0.05
    xOld = 0
    locations2 = [0]
    # Incorrect markov chain using SHO Energy
    for i in range(100000):
        deltaX = st.norm.rvs(loc=0, scale=1)
        acceptanceRatio = np.exp(-(deltaX**2 + (2*xOld*(deltaX)))/2)
        num = np.random.rand(1)
        if num[0] < acceptanceRatio:
            xOld = xOld + deltaX
        locations2.append(xOld)

    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    ax2.hist(locations2[1000:], bins=100, density=True, label="Experimental Result")
    ax2.plot(xVals, np.exp((-xVals**2)/2)/np.sqrt(2*np.pi), zorder=100, label="Theorectical Distribution")
    ax2.legend(loc="upper left")
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Location")
    fig2.suptitle("Probability distribution of locations \n using MCMC with SHO energy")
    #fig2.savefig("Plots/Fast Diffusion Approach 2 SHO")

    # Approach 2 Langevin Equation
    delta = 0.05
    xOld = 0
    locations2 = [0]
    acceptanceRatios = []
    # Markov chain using SHO Energy
    for i in range(100000):
        deltaX = st.norm.rvs(loc=0, scale=delta)
        xNew = xOld + (delta/2)*(-xOld) + np.sqrt(delta)*deltaX
        # Ratio of configurations
        acceptanceRatio = np.exp(-(deltaX**2 + (2*xOld*(deltaX)))/2) 
        # Ratio of transitions
        acceptanceRatio *= np.exp(delta*(xOld - xNew)*(-xOld-xNew) + ((delta/2)**2)*(xOld**2 - xNew**2))
        acceptanceRatios.append(acceptanceRatio)
        locations2.append(xNew)
    # Checking out the stats
    print("The acceptance Ratio for the Langevin Equation is:")
    print(np.mean(acceptanceRatios[1000:]), u"\u00B1", np.std(acceptanceRatios[1000:]))

# Doing Faster Diffusion and Annalysis
def fasterDiffusion(timesteps):
    betas = np.linspace(0.0001, 0.02, timesteps, dtype=np.float32)
    finalLocationArray = []
    xRunArray = []
    # This loop is for the final locations Histrogram
    for i in range(10000):
        finalLocation, _ = ForwardDiffusion2(0, timesteps, betas)
        finalLocationArray.append(finalLocation)
    finalLocationArray = np.array(finalLocationArray)

    # This loop is for plotting x runs
    for j in range(5):
        _, xlocations = ForwardDiffusion2(0, timesteps, betas)
        xRunArray.append(xlocations) 


    # xVals for theory
    xVals = np.linspace(-3, 3, 1001)

    #Plotting
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax[0].hist(finalLocationArray, bins=100, density=True, label="Experimental Result")
    ax[0].plot(xVals, np.exp((-xVals**2)/2)/np.sqrt(2*np.pi), zorder=100, label="Theorectical Distribution")
    ax[0].legend(loc="upper left")
    ax[0].set_ylabel("Probability")
    ax[0].set_xlabel("Location")
    ax[0].set_title("Final Locations Histogram")
    for j in range(len(xRunArray)):
        ax[1].plot(np.arange(timesteps+1), xRunArray[j], label="Run {}".format(j+1))
    ax[1].set_ylabel("Position")
    ax[1].set_xlabel("Timestep")
    ax[1].set_title("Variation across different runs")
    ax[1].legend(loc="upper left")
    fig.suptitle("Time Dependant Langevin Dynamics")
    fig.tight_layout()
    #fig.savefig("Plots/Time Dependant Langevin Dynamics")
    
    # Now we do Time dependant Langevin Dynamics but quicker
    alphas = 1 - betas
    alphaBarTs = np.cumprod(alphas)
    finalLocationArray2 = []
    for i in range(10000):
        xlocations = ForwardDiffusionFast(0, alphaBarTs)
        finalLocationArray2.append(xlocations[-1])
    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    
    ax2.hist(finalLocationArray2, bins=100, density=True, label="Experimental Result")
    ax2.plot(xVals, np.exp((-xVals**2)/2)/np.sqrt(2*np.pi), zorder=100, label="Theorectical Distribution")
    ax2.legend(loc="upper left")
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Location")
    ax2.set_title("Fast Forward Diffusion location distribution")
    #fig2.savefig("Plots/Fast Forward Diffusion Histogram")
    plt.show()

# Works our the plots for undiffusion
def undiffusion(timesteps):
    betas = np.linspace(0.0001, 0.02, timesteps, dtype=np.float32)
    alphaBarts = np.cumprod(1 - betas)

    #Add code to unpickle neural net and  make a new guess z function
    file = open("DiffusionNeuralNet.obj", 'rb')
    #file = open("DiffusionNeuralNetPrompt.obj", 'rb')
    NeuralNet = pickle.load(file)
    file.close()

    # Running undiffusion for histogram
    initPositions = []
    forwardDiffusionPositions = []
    undiffusionPositions = []
    for j in range(10000):
        print(f'\r{j}', end='')
        locations = ForwardDiffusionFast(0, alphaBarts)
        locationsBack = undiffuse(locations[-1], betas, timesteps, NeuralNet)
        initPositions.append(locationsBack[0])
        forwardDiffusionPositions.append(locations[124])
        undiffusionPositions.append(locationsBack[124])

    # Running undiffuion for position vs. time plot
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for i in range(5):
        locations = ForwardDiffusionFast(0, alphaBarts)
        locationsBack = undiffuse(locations[-1], betas, timesteps, NeuralNet)
        ax[0].plot(np.arange(timesteps), locationsBack, label="Undiffusion Run {}".format(i+1))
    ax[0].legend()
    ax[0].set_ylabel("Position")
    ax[0].set_xlabel("Time")
    ax[0].set_title("Multiple runs over time")
    ax[1].hist(initPositions)
    ax[1].set_title("Initial Position Histogram")
    ax[2].hist(forwardDiffusionPositions, density=True, bins=50, alpha=0.5, label="Forward Diffusion", zorder=200)
    ax[2].hist(undiffusionPositions, density=True, bins=50, label="Undiffusion")
    ax[2].set_title("Forward Diffusion and Undiffusion \n at T=125")
    ax[2].legend()
    fig.suptitle("Undiffusion Dynamics Neural Network")
    fig.savefig("Plots/Undiffusion Dynamics Neural Network")
    plt.show()

def undiffusionTraining(timesteps):
    # Time-dependant betas
    betas = np.linspace(0.0001, 0.02, timesteps, dtype=np.float32)
    alphaBarts = np.cumprod(1 - betas)
    # Defining neural net parameters
    #numInput, numHidden, numOut = 2, 15, 1
    numInput, numHidden, numOut = 3, 15, 1

    neuralNet = nn.Sequential(nn.Linear(numInput, numHidden), nn.ReLU(), nn.Linear(numHidden, numHidden), nn.ReLU(), nn.Linear(numHidden, numOut))
    #neuralNet(torch.Tensor([3.0, 4]))

    # Pytorch Optimization
    lossFunction = nn.MSELoss()
    opt = torch.optim.Adam(neuralNet.parameters(), lr=1e-3)

    lossList, window = [], []
    for step in range(200000):
        opt.zero_grad()
        # Choose random position and t
        x0 = sampleFromPInit()
        t = np.random.randint(1, timesteps)

        # get noisyData and noise from ForwardDiffusionFast
        noise, noisyData = ForwardDiffusionFastWithNoise(x0, alphaBarts[:t])
        noisyData = np.array(noisyData)[t-1]

        prompt = -1 if x0 < 0 else 1

        #noisyData = torch.tensor([noisyData, t]).float()
        noisyData = torch.tensor([noisyData, t, prompt]).float()
        noise = torch.tensor([noise]).float()

        loss = lossFunction(noise, neuralNet(noisyData))
        loss.backward()
        opt.step()

        print(f'\r{step}', end='')

        # Stroing loss in list with window averaging
        window.append(loss.item())
        if (step%100 == 0):
            lossList.append(np.mean(window))
            window = []

    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.arange(len(lossList)), lossList)
    ax.set_yscale("log")
    ax.set_title("Neural Net loss over time")
    ax.set_xlabel("Time per hundred steps")
    ax.set_ylabel("Loss")
    fig.savefig("Plots/NeuralNetLoss")
    plt.show()

    # Pickling neural net
    #file = open("DiffusionNeuralNet.obj", "wb")
    file = open("DiffusionNeuralNetPrompt.obj", "wb")
    pickle.dump(neuralNet, file)
    file.close()


if __name__ == "__main__":
    #diffusionGraphs()
    #fasterDiffusion(200)
    undiffusion(200)
    #undiffusionTraining(200)