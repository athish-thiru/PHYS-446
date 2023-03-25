import numpy as np
import matplotlib.pyplot as plt

# RBM Class
class RestrictedBoltzmannMachine:
    # Initialize Class
    def __init__(self, visibleSize, hiddenSize):
        self.visibleSize = visibleSize
        self.hiddenSize = hiddenSize
        # The states are either 1 or -1
        self.visibleStates = np.random.choice([-1, 1], visibleSize)
        self.hiddenStates = np.random.choice([-1, 1], hiddenSize)
        # biases take any decimal for -1 to 1
        self.visibleBiases = np.random.default_rng().uniform(-1, 1, visibleSize)
        self.hiddenBiases = np.random.default_rng().uniform(-1, 1, hiddenSize)
        # Weights for interaction of visible and hidden states
        self.weights = np.random.default_rng().uniform(-1, 1, (visibleSize, hiddenSize))
    
    # calculates energy of RBM for a particular set of visibleStates and hiddenStates
    def calculateEnergy(self):
        energy1 = - (self.visibleBiases@self.visibleStates.T) - (self.hiddenBiases@self.hiddenStates.T)
        energy2 = - (self.visibleStates@self.weights@self.hiddenStates.T)
        return energy1 + energy2
    
    # Updating Hidden Layer
    def updateHiddenLayer(self):
        # Probability for hidden layers
        mh = self.visibleStates@self.weights + self.hiddenBiases
        ph = np.exp(mh)/(np.exp(mh) + np.exp(-mh))
        randh = np.random.rand(ph.size)
        hiddenSpinUpMask = randh < ph
        hiddenSpinDownMask = randh > ph
        self.hiddenStates[hiddenSpinUpMask] = 1
        self.hiddenStates[hiddenSpinDownMask] = -1
    
    # Updating Visible Layer
    def updateVisibleLayer(self):
        # Probability for visible layers
        ms = self.weights@self.hiddenStates.T + self.visibleBiases
        ps = np.exp(ms)/(np.exp(ms) + np.exp(-ms))
        rands = np.random.rand(ps.size)
        visibleSpinUpMask = rands < ps
        visibleSpinDownMask = rands > ps
        self.visibleStates[visibleSpinUpMask] = 1
        self.visibleStates[visibleSpinDownMask] = -1

    # Performs Gibbs Sampling
    def gibbsSampling(self, k):
        for i in range(k):
            self.updateHiddenLayer()
            self.updateVisibleLayer()
    
    # Get Hidden States
    def getHiddenStates(self):
        return self.hiddenStates
    
    # Set Hidden States
    def setHiddenStates(self, hiddenStates):
        self.hiddenStates = hiddenStates

    # Gets Visible States
    def getVisibleStates(self):
        return self.visibleStates
    
    # Sets Visible States
    def setVisibleStates(self, visibleStates):
        self.visibleStates = visibleStates

# Plots the joint and marginal distribution of the possible hidden and visible states
# visibleLayerNum is the size of the visible layer
# hiddenLayerNum is the size of the hidden layer
def plotJointandMarginalProbabilityDistributions(visibleLayerNum, hiddenLayerNum):
    # Array with powers of two for easier calculations
    powersOfTwo = []
    for i in range(visibleLayerNum+hiddenLayerNum):
        powersOfTwo.append(2**(visibleLayerNum+hiddenLayerNum-i-1))
    powersOfTwo = np.array(powersOfTwo)

    # Creating RBM
    rbm = RestrictedBoltzmannMachine(visibleLayerNum, hiddenLayerNum)
    
    # MCMC through Gibbs Sampling to obtain stationary distributions
    jointStates, visibleStates, hiddenStates = [], [], []
    actualJointProbs, actualVisibleProbs, actualHiddenProbs = [], [], []
    for i in range(100000):
        rbm.gibbsSampling(10) # taking k=10 to make MCMC chains stationary
        visibleState, hiddenState = rbm.getVisibleStates(), rbm.getHiddenStates()
        currentState = np.concatenate((visibleState, hiddenState))
        visibleStates.append(np.sum(np.where(visibleState == -1, 0, 1)*powersOfTwo[-visibleLayerNum:]))
        hiddenStates.append(np.sum(np.where(hiddenState == -1, 0, 1)*powersOfTwo[-hiddenLayerNum:]))
        jointStates.append(np.sum(np.where(currentState == -1, 0, 1)*powersOfTwo))
    
    # Calculating theoretical values for p(v,h)
    for j in range(2**(visibleLayerNum+hiddenLayerNum)):
        binRep = str(bin(j))[2:]
        while (len(binRep) != (visibleLayerNum+hiddenLayerNum)):
            binRep = "0" + binRep
        
        initState = np.where(np.array([int(elem) for elem in binRep]) == 0, -1, 1)
        rbm.setVisibleStates(initState[:visibleLayerNum])
        rbm.setHiddenStates(initState[-hiddenLayerNum:])
        energy = rbm.calculateEnergy()
        probability = np.exp(-energy)
        actualJointProbs.append(probability)
    actualJointProbs = actualJointProbs/np.sum(actualJointProbs)
    
    # Calculating Theoretical value for p(v)
    for j in range(2**(visibleLayerNum)):
        vBinRep = str(bin(j))[2:]
        while (len(vBinRep) != (visibleLayerNum)):
            vBinRep = "0" + vBinRep
        
        vInitState = np.where(np.array([int(elem) for elem in vBinRep]) == 0, -1, 1)
        rbm.setVisibleStates(vInitState)
        probability = 0
        for k in range(2**(hiddenLayerNum)):
            hBinRep = str(bin(k))[2:]
            while (len(hBinRep) != (hiddenLayerNum)):
                hBinRep = "0" + hBinRep
            hInitState = np.where(np.array([int(elem) for elem in hBinRep]) == 0, -1, 1)
            rbm.setHiddenStates(hInitState)
            energy = rbm.calculateEnergy()
            probability += np.exp(-energy)
        actualVisibleProbs.append(probability)
    actualVisibleProbs = actualVisibleProbs/np.sum(actualVisibleProbs)

    # Calculating Theoretical value for p(h)
    for k in range(2**(hiddenLayerNum)):
        hBinRep = str(bin(k))[2:]
        while (len(hBinRep) != (hiddenLayerNum)):
            hBinRep = "0" + hBinRep
        hInitState = np.where(np.array([int(elem) for elem in hBinRep]) == 0, -1, 1)
        rbm.setHiddenStates(hInitState)
        probability = 0

        for j in range(2**(visibleLayerNum)):
            vBinRep = str(bin(j))[2:]
            while (len(vBinRep) != (visibleLayerNum)):
                vBinRep = "0" + vBinRep
            
            vInitState = np.where(np.array([int(elem) for elem in vBinRep]) == 0, -1, 1)
            rbm.setVisibleStates(vInitState)
            energy = rbm.calculateEnergy()
            probability += np.exp(-energy)
        actualHiddenProbs.append(probability)
    actualHiddenProbs = actualHiddenProbs/np.sum(actualHiddenProbs)

    # Plotting joint and marginal probabilities
    jHist, _ = np.histogram(jointStates, bins=2**(visibleLayerNum+hiddenLayerNum))
    vHist, _ = np.histogram(visibleStates, bins=2**(visibleLayerNum))
    hHist, _ = np.histogram(hiddenStates, bins=2**(hiddenLayerNum))
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    ax[0].step(np.arange(2**(visibleLayerNum+hiddenLayerNum)), jHist/np.sum(jHist), where="mid", label="MCMC Approximation")
    ax[0].plot(np.arange(2**(visibleLayerNum+hiddenLayerNum)), actualJointProbs, "rx", zorder=100, label="Actual Value")
    ax[0].set_xlabel("States")
    ax[0].set_ylabel("Probability")
    ax[0].set_title("p(v,h)")
    ax[0].legend()
    ax[1].step(np.arange(2**visibleLayerNum), vHist/np.sum(vHist), where="mid", label="MCMC Approximation")
    ax[1].plot(np.arange(2**visibleLayerNum), actualVisibleProbs, "rx", zorder=100, label="Actual Value")
    ax[1].set_xlabel("States")
    ax[1].set_ylabel("Probability")
    ax[1].set_title("p(v)")
    ax[1].legend()
    ax[2].step(np.arange(2**hiddenLayerNum), hHist/np.sum(hHist), where="mid", label="MCMC Approximation")
    ax[2].plot(np.arange(2**hiddenLayerNum), actualHiddenProbs, "rx", zorder=100, label="Actual Value")
    ax[2].set_xlabel("States")
    ax[2].set_ylabel("Probability")
    ax[2].set_title("p(h)")
    ax[2].legend()
    fig.suptitle("Joint and Marginal Probability Distribution for \n Restricted Boltzmann Machines")
    fig.tight_layout()
    fig.savefig("Plots/Joint and Marginal Probability Distribution for RBM")
    plt.show()

# Plots the conditional distribution of the possible hidden and visible states
# visibleLayerNum is the size of the visible layer
# hiddenLayerNum is the size of the hidden layer
def plotConditionalProbabilityDistribution(visibleLayerNum, hiddenLayerNum):
    # Array with powers of two for easier calculations
    powersOfTwo = []
    for i in range(visibleLayerNum+hiddenLayerNum):
        powersOfTwo.append(2**(visibleLayerNum+hiddenLayerNum-i-1))
    powersOfTwo = np.array(powersOfTwo)

    # Dictionary to count the distribution of each state
    distributionDict = {}

    # Creating RBM
    rbm = RestrictedBoltzmannMachine(visibleLayerNum, hiddenLayerNum)

    # MCMC through Gibbs Sampling to obtain stationary distributions
    for i in range(1000000):
        rbm.gibbsSampling(10) # taking k=10 to make MCMC chains stationary
        visibleState, hiddenState = rbm.getVisibleStates(), rbm.getHiddenStates()
        visibleIntState = np.sum(np.where(visibleState == -1, 0, 1)*powersOfTwo[-visibleLayerNum:])
        hiddenIntState = np.sum(np.where(hiddenState == -1, 0, 1)*powersOfTwo[-hiddenLayerNum:])

        # Checking if key in dictionary otherwise creating it
        if (visibleIntState, hiddenIntState) in distributionDict:
            distributionDict[(visibleIntState, hiddenIntState)] += 1
        else:
            distributionDict[(visibleIntState, hiddenIntState)] = 1

    
    vGivenHHist, hGivenVHist = [], []
    vGivenHTheory, hGivenVTheory = [], []
    for i in range(2**(visibleLayerNum)):
        for j in range(2**(hiddenLayerNum)):
            # Print out the visible state, hidden state and joint state and found out they end up mapping like
            # Joint = (2**(hiddenLayerNum))*Visible + Hidden
            # If probability exists append that otherwise append 0
            if (i, j) in distributionDict:
                vGivenHHist.append(distributionDict[(i, j)])
                hGivenVHist.append(distributionDict[(i, j)])
            else:
                vGivenHHist.append(0)
                hGivenVHist.append(0)
            
            # Calculating theoretical values
            vBinRep = str(bin(i))[2:]
            while (len(vBinRep) != (visibleLayerNum)):
                vBinRep = "0" + vBinRep
            vState = np.where(np.array([int(elem) for elem in vBinRep]) == 0, -1, 1)
        
            hBinRep = str(bin(j))[2:]
            while (len(hBinRep) != (hiddenLayerNum)):
                hBinRep = "0" + hBinRep
            hState = np.where(np.array([int(elem) for elem in hBinRep]) == 0, -1, 1)

            rbm.setVisibleStates(vState)
            rbm.setHiddenStates(hState)
            probability = np.exp(-rbm.calculateEnergy())
            vGivenHTheory.append(probability)
            hGivenVTheory.append(probability)

    # Convert to numpy array
    vGivenHHist, hGivenVHist = np.array(vGivenHHist, dtype=float), np.array(hGivenVHist, dtype=float)
    vGivenHTheory, hGivenVTheory = np.array(vGivenHTheory), np.array(hGivenVTheory)
    
    # Dividing by the appropriate sum of states for theory and experiment
    for i in range(2**(hiddenLayerNum)):
        vGivenHHist[i::2**(hiddenLayerNum)] = vGivenHHist[i::2**(hiddenLayerNum)]/np.sum(vGivenHHist[i::2**(hiddenLayerNum)])
        vGivenHTheory[i::2**(hiddenLayerNum)] = vGivenHTheory[i::2**(hiddenLayerNum)]/np.sum(vGivenHTheory[i::2**(hiddenLayerNum)])
    
    for j in range(2**(visibleLayerNum)):
        idx = j*(2**(hiddenLayerNum))
        hGivenVHist[idx:idx + 2**(hiddenLayerNum)] = hGivenVHist[idx:idx + 2**(hiddenLayerNum)]/np.sum(hGivenVHist[idx:idx + 2**(hiddenLayerNum)])
        hGivenVTheory[idx:idx + 2**(hiddenLayerNum)] = hGivenVTheory[idx:idx + 2**(hiddenLayerNum)]/np.sum(hGivenVTheory[idx:idx + 2**(hiddenLayerNum)])

    # Plotting
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].step(np.arange(2**(visibleLayerNum+hiddenLayerNum)), vGivenHHist, where="mid", label="MCMC Approximation")
    ax[0].plot(np.arange(2**(visibleLayerNum+hiddenLayerNum)), vGivenHTheory, "rx", zorder=100, label="Actual Value")
    ax[0].set_xlabel("States")
    ax[0].set_ylabel("Probability")
    ax[0].set_title("p(v|h)")
    ax[0].legend()
    ax[1].step(np.arange(2**(visibleLayerNum+hiddenLayerNum)), hGivenVHist, where="mid", label="MCMC Approximation")
    ax[1].plot(np.arange(2**(visibleLayerNum+hiddenLayerNum)), hGivenVTheory, "rx", zorder=100, label="Actual Value")
    ax[1].set_xlabel("States")
    ax[1].set_ylabel("Probability")
    ax[1].set_title("p(h|v)")
    ax[1].legend()
    fig.suptitle("Conditional Probability Distribution for \n Restricted Boltzmann Machines")
    fig.tight_layout()
    fig.savefig("Plots/Conditional Probability Distribution for RBM")
    plt.show()


if __name__ == "__main__":
    #plotJointandMarginalProbabilityDistributions(5, 2)
    plotConditionalProbabilityDistribution(5, 2)
